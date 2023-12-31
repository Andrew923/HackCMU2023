CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230421000000_e20230421235959_p20230422030555_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-04-22T03:05:55.652Z   date_calibration_data_updated         2023-04-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-04-21T00:00:00.000Z   time_coverage_end         2023-04-21T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxz�   �          @Ϯ@3�
�H��@�{BF{C���@3�
���@0  A��HC�E                                    BxzӦ  �          @�  @.�R�J�H@�
=BG�\C�g�@.�R��ff@1G�A�  C�޸                                    Bxz�L  
�          @θR@!��S�
@�BG\)C���@!�����@*�HA�C��                                     Bxz��  
�          @�{@\)�N{@�\)BK  C��f@\)��  @0��A�\)C��
                                    Bxz��  �          @�{@$z��L��@�ffBIQ�C�g�@$z���
=@/\)A�  C�!H                                    Bxz>  
�          @�(�@1G��P��@��B?��C�(�@1G���p�@!�A��\C�R                                    Bxz�  T          @�z�@8���Q�@��B;G�C��3@8����z�@��A���C���                                    Bxz+�  
�          @�(�@6ff�W
=@��B9=qC�&f@6ff��{@�A��RC�b�                                    Bxz:0  T          @�33@W��r�\@qG�B�\C��3@W�����?�
=AQ��C���                                    BxzH�  
�          @�z�@]p��w�@l(�B�\C�� @]p���?���A?\)C�ٚ                                    BxzW|  �          @���@^{�s33@qG�B(�C�33@^{���?�
=AO�
C��{                                    Bxzf"  �          @�z�@O\)�l(�@�=qB�\C���@O\)��ff?޸RA~{C��
                                    Bxzt�  T          @˅@W��n{@x��B(�C�
@W�����?�=qAe�C���                                    Bxz�n  �          @ʏ\@Y����  @a�B��C�,�@Y����
=?�\)A#�C���                                    Bxz�  
�          @��H@Y���p  @tz�BffC�)@Y����z�?�G�A\(�C���                                    Bxz��  
�          @ʏ\@u�z�H@G�A�
=C�0�@u��ff?O\)@��
C��                                    Bxz�`  T          @��H@L(��|��@qG�BffC�s3@L(���G�?�{AG33C��=                                    Bxz�  T          @ʏ\@r�\�xQ�@P��A�C�#�@r�\��\)?s33A	�C��H                                    Bxz̬  
�          @��@i���u@Z=qB�C�Ǯ@i������?�{A!�C�f                                    Bxz�R  �          @��H@fff�xQ�@^�RB33C�l�@fff���H?�33A&�RC���                                    Bxz��  �          @ʏ\@e�w�@_\)BC�n@e���\?�z�A)�C���                                    Bxz��  T          @��H@g��~{@W
=B   C�4{@g���33?�  A��C��3                                    BxzD  "          @�z�@^�R��  @c�
Bz�C�� @^�R��\)?�z�A'�
C���                                    Bxz�  T          @�(�@HQ��|��@z�HBffC�8R@HQ����?�  AY�C�+�                                    Bxz$�  "          @�@E�\)@~�RB
=C��H@E��p�?��A]C��
                                    Bxz36  T          @��
@A��}p�@~{B33C���@A���z�?�ffA`��C��=                                    BxzA�  
�          @˅@AG���  @z=qB�
C���@AG���z�?�p�AW
=C��                                     BxzP�  �          @��
@>�R���@qG�BffC�˅@>�R��
=?��\A9�C�J=                                    Bxz_(  T          @�(�@G����\@q�Bz�C��
@G�����?�=qA@z�C��                                    Bxzm�  "          @�(�@P  �{�@u�B��C�˅@P  ��G�?�Q�AQp�C��f                                    Bxz|t  �          @��
@N�R��G�@n{B  C�L�@N�R���H?�ffA<(�C���                                    Bxz�  
�          @��@J=q��z�@n{B��C���@J=q��p�?�  A3�C�"�                                    Bxz��  "          @��@A���  @\)B�C��@A���?�ffA`z�C���                                    Bxz�f  
�          @�(�@B�\��  @|(�BffC��R@B�\���?�G�A[33C���                                    Bxz�  �          @�z�@Fff��  @y��BffC��q@Fff��z�?�p�AU�C���                                    BxzŲ  �          @�ff@7
=���R@|(�B  C�R@7
=���H?�AK�C���                                    Bxz�X  �          @�{@9����33@�G�BG�C���@9����G�?ǮA`(�C��{                                    Bxz��  
�          @�@9�����H@���B��C��=@9������?�ffA^�RC���                                    Bxz�  T          @Ϯ@G���{@vffBp�C�Q�@G�����?���A@  C��
                                    Bxz J  
Z          @�
=@XQ��u@|��B��C���@XQ�����?�{Ag33C�S3                                    Bxz�  
�          @�p�@P  �|(�@y��BQ�C���@P  ���\?\AZ�\C���                                    Bxz�  
Z          @�@a��c33@���BQ�C�p�@a�����?���A��C�}q                                    Bxz,<  "          @�@_\)�B�\@�B.�C�s3@_\)����@�HA�p�C��                                    Bxz:�  �          @θR@[��J�H@�p�B-�RC���@[�����@
=A��C��H                                    BxzI�  �          @�  @h���7�@�G�B1�\C��)@h����ff@&ffA�
=C��H                                    BxzX.  
�          @У�@^{�p  @���B��C�c�@^{��
=?�(�AuC�Ф                                    Bxzf�  T          @�  @[��xQ�@z�HBffC��R@[�����?���A`z�C�|)                                    Bxzuz  
�          @�ff@Z=q�u@y��B�RC��@Z=q���?�=qAb�RC���                                    Bxz�   �          @�ff@X���p  @�  BG�C��@X�����R?��HAv=qC��f                                    Bxz��  
�          @�\)@QG��xQ�@���B�\C��@QG����H?�Ao\)C��H                                    Bxz�l  �          @�  @X������@p��B�C��q@X����33?�\)AB�HC�.                                    Bxz�  T          @�  @Z=q�u�@\)BffC��R@Z=q����?�An{C�u�                                    Bxz��  
�          @�ff@W�����@n{BG�C��R@W����?��A@��C�1�                                    Bxz�^  �          @�ff@Tz��vff@}p�B��C�aH@Tz�����?��Ak�C�R                                    Bxz�  T          @�z�@QG�����@\(�BffC��
@QG�����?z�HA�C��)                                    Bxz�  
�          @�z�@+���=q@O\)A�G�C��@+�����?��@���C�l�                                    Bxz�P  z          @�@Dz�����@@  Aޏ\C�W
@Dz���(�>�Q�@L(�C�H�                                    Bxz�  �          @�ff@`  ���H@7�A�G�C��@`  ����>�{@?\)C�|)                                    Bxz�  �          @�Q�@\(����H@Z�HA�p�C�4{@\(���ff?p��A��C�&f                                    Bxz%B  �          @ҏ\@`����z�@[�A��C�Y�@`�����?n{A{C�P�                                    Bxz3�  �          @У�@XQ����H@^�RBG�C���@XQ���
=?�  AG�C���                                    BxzB�  
�          @��@W�����@j=qB��C�*=@W����?���A((�C��f                                    BxzQ4  T          @�z�@S�
��Q�@e�B�\C�)@S�
��?�G�AQ�C�q                                    Bxz_�  T          @��H@Q���33@mp�B�C�� @Q����H?���A(  C�33                                    Bxzn�  "          @�(�@XQ����R@uB�\C�^�@XQ�����?���A@��C��R                                    Bxz}&  
�          @�p�@W���@|��BffC�q�@W�����?�  APQ�C���                                    Bxz��  �          @�ff@W���  @z=qBG�C�,�@W���33?�Q�AFffC��H                                    Bxz�r  "          @�{@Q�����@|(�B(�C��=@Q�����?�Q�AF=qC��                                    Bxz�  �          @�
=@S33��Q�@���B��C���@S33����?��
AR�HC�#�                                    Bxz��  
�          @ָR@Tz���\)@�  B�C��@Tz����
?��
AS�C�K�                                    Bxz�d  .          @�@S33��=q@�z�B=qC���@S33��G�?�p�Ap(�C�b�                                    Bxz�
            @Ӆ@U�����@���B�
C���@U���
=?У�Ae�C���                                    Bxz�  �          @�(�@N�R��
=@k�B�HC��3@N�R��p�?��A�HC��
                                    Bxz�V  �          @�p�@R�\���R@k�B=qC�4{@R�\��p�?�33A�\C��                                    Bxz �  �          @��
@W���=q@\)B33C��@W���
=?�{A`��C���                                    Bxz�  
�          @��
@Vff����@i��B�C��q@Vff��33?�33A Q�C�l�                                    BxzH  .          @�z�@Z=q�\)@��B�RC�=q@Z=q��?��HAnffC�                                    Bxz,�  
�          @��H@X����33@y��B  C��=@X����ff?\AUp�C��\                                    Bxz;�  `          @�G�@p�����?���A9�C��q@p�������\�UC��
                                    BxzJ:  z          @�?�33��=q=�\)?J=qC�G�?�33��Q���
���
C�k�                                    BxzX�  T          @�z�@=q���?��A<��C�@=q���
����UG�C�#�                                    Bxzg�  
�          @��@333��{?�\)A�G�C�k�@333���׿
=q��33C��3                                    Bxzv,  �          @��@�
����?c�
A�RC�]q@�
��33�Ǯ���C��)                                    Bxz��  �          @�Q�@���{?+�@���C�e@������H����C��q                                    Bxz�x  �          @���@�
���>�=q@1�C��
@�
��z������33C���                                    Bxz�  
�          @���@*�H��G�?O\)A ��C��3@*�H���H�У����
C�s3                                    Bxz��  �          @�\)@@����Q�?�p�A=��C�~�@@����Q쿠  �@��C���                                    Bxz�j  �          @��R@<(���=q?�=qA&�\C�{@<(���\)��33�Z{C�G�                                    Bxz�  �          @�{@G
=��ff?��A!G�C�@G
=���
�����W33C�G�                                    Bxzܶ  
�          @�@Dz���\)?z�HA\)C�Ф@Dz����
�����bffC��                                    Bxz�\  T          @�(�@6ff���\?G�@��HC���@6ff�������  C�4{                                    Bxz�  
�          @�=q@8Q���  ?@  @�ffC���@8Q����ÿ�33���C���                                    Bxz�  �          @���@:�H��
=?0��@ٙ�C�4{@:�H��
=��Q����C�ٚ                                    BxzN  
�          @�Q�@7���
=?�\@��
C��)@7���z����G�C��
                                    Bxz%�  �          @��@'
=��Q�>aG�@��C���@'
=��������{C��                                    Bxz4�  �          @���@(���33�L�;��C�Ǯ@(���  ����ȣ�C�9�                                    BxzC@  T          @�p�@ �����\=L��?   C�+�@ �������33���C��                                    BxzQ�  
(          @�z�@����\�8Q��ffC�˅@����   �ҏ\C�ff                                    Bxz`�  
�          @�
=@*�H����>.{?�  C���@*�H����
�H���\C�.                                    Bxzo2  �          @��@+���  �8Q��C�R@+���33�p��ͅC�˅                                    Bxz}�  
�          @���@ff��������EC���@ff�����#�
�ܸRC�Z�                                    Bxz�~            @�\)@���
=��R��Q�C�e@��z=q�-p����C���                                    Bxz�$  T          @���@G
=��=q>�z�@O\)C��@G
=�p  ��{��Q�C��3                                    Bxz��  
�          @�(�@����`  ?��A3�C�W
@����g
=�!G���C��                                    Bxz�p  
�          @�z�@�G��c�
?Q�A�C�(�@�G��b�\�h���  C�>�                                    Bxz�  �          @���@�G��p  ?��@�ffC��{@�G��e���p��L��C�7
                                    Bxzռ  �          @�33@b�\��Q�\)���RC���@b�\�`�׿��H���C��H                                    Bxz�b  �          @�
=@e�����(����
=C��=@e�S33�(���z�C��
                                    Bxz�  
�          @�
=@s33�\(��������\C��f@s33�{�N{���C���                                    Bxz�  "          @���@]p��A��#33��Q�C�^�@]p�����i���0
=C�                                    BxzT  
�          @�z�?=p����R������C��\?=p��H���h���A�HC���                                    Bxz�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxz-�   &          @�G��}p��������qCr^��}p�>\��p���C��                                    Bxz<F  "          @���?E���  �����C�!H?E��H���L���2�RC��                                    BxzJ�  �          @�  @   �e�@N�RB��C�u�@   ���?�  AU�C��                                    BxzY�  "          @�@"�\�{�@J=qB{C�p�@"�\��{?��A(��C���                                    Bxzh8  "          @�G�@�����@^�RB�HC�33@���p�?��
AM��C�n                                    Bxzv�  �          @�=q@�s�
@vffB&\)C�c�@��z�?�(�A���C��                                    Bxz��  
�          @���@(��tz�@vffB$��C���@(���z�?��HA��C��                                    Bxz�*  
�          @�
=?�{�P  @���BL�C���?�{��  @+�A�C�33                                    Bxz��  
�          @�33?�Q���@O\)B	Q�C�u�?�Q����?c�
A\)C��\                                    Bxz�v  
�          @���@=q�0  @�G�BR��C��q@=q���@B�\A�Q�C��=                                    Bxz�  "          @���@!��2�\@�\)BN
=C�q@!����
@>{A�C�Z�                                    Bxz��  
�          @\@"�\�#33@���BV�
C�w
@"�\��
=@N�RB z�C��=                                    Bxz�h  �          @\@#�
�0��@���BN��C�u�@#�
��33@AG�A�RC��                                    Bxz�  �          @�(�@&ff�(Q�@�(�BSffC�T{@&ff��G�@K�A��\C��                                    Bxz��  �          @�@(���'�@�BT  C���@(������@O\)A���C��                                    Bxz	Z  �          @ƸR@��Q�@�{Bc�C��H@���ff@e�B��C�H�                                    Bxz   "          @�\)@p��У�@��\B(�C��{@p��|��@���B-�C��H                                    Bxz&�  T          @�Q�?�(���z�@�ffB�W
C�H?�(�����@��B0��C�\                                    Bxz5L  T          @�  @
=q��@�=qB|{C��=@
=q��(�@��B'(�C���                                    BxzC�  �          @�G�@���
=@�
=Bq�HC��)@���p�@�G�B�
C��                                    BxzR�  T          @ʏ\@����33@�G�BtG�C��f@����p�@��B"�C��3                                    Bxza>  T          @��H?����=q@�B�G�C�  ?������@��B;=qC�b�                                    Bxzo�  T          @��H?�
=�˅@���B�.C�L�?�
=����@��\B9(�C�H�                                    Bxz~�  
�          @˅?ٙ���
=@�z�B��)C��R?ٙ����
@�G�B6�C�<)                                    Bxz�0  �          @�33?�(���@��HB���C�Ǯ?�(����R@�{B1G�C��                                    Bxz��  �          @ʏ\@���\)@�  Br\)C���@����ff@|(�BffC���                                    Bxz�|  �          @�=q@#33��R@��RB^C��@#33����@e�B
=qC���                                    Bxz�"  �          @ʏ\@�H��H@���Bd�RC�y�@�H��Q�@l(�B�C�{                                    Bxz��  T          @�=q@33�Q�@�(�Bi�RC��q@33��Q�@q�B{C�~�                                    Bxz�n  T          @�=q@��+�@�{B\\)C�
@���ff@^�RBG�C���                                    Bxz�  �          @���@ff�p�@�(�Bk��C�1�@ff���\@p  B��C�K�                                    Bxz�  T          @ʏ\@ ���*�H@��
Bh\)C�z�@ ����Q�@i��B�C�s3                                    Bxz`  �          @��H@��4z�@�G�Ba�HC�,�@����@`��Bz�C��3                                    Bxz  
�          @�=q@���J=q@���BR(�C�k�@����G�@HQ�A��C���                                    Bxz�  
�          @��H@
�H�<(�@�B[�C�<)@
�H���@W�B ffC��\                                    Bxz.R  �          @�=q@��QG�@�\)BPz�C�@ @����
@C33A�33C��                                    Bxz<�  
�          @ʏ\@z��?\)@�B[��C�U�@z���ff@VffB   C�T{                                    BxzK�  T          @��H@�E�@�(�BX�C�{@��Q�@QG�A��\C�Q�                                    BxzZD  �          @��H@
=�Fff@��BW{C�&f@
=��Q�@P  A���C�j=                                    Bxzh�  T          @�33@���E�@��BV
=C��f@����  @P��A���C��                                     Bxzw�  
�          @��H@
=q�L(�@�G�BRp�C��@
=q���@I��A�Q�C���                                    Bxz�6  �          @ə�@���=p�@��BX�
C�N@����(�@Tz�A�Q�C�q                                    Bxz��  T          @�G�@33�A�@��
BY�RC��@33��ff@S33A��RC�<)                                    Bxz��  
�          @��@��Mp�@�  BRQ�C�y�@����@G�A�G�C�,�                                    Bxz�(  T          @�=q@G��e�@���BEC���@G����@1�A���C�p�                                    Bxz��  
�          @��H@+��%�@�Q�BUC�f@+����@Y��BG�C�c�                                    Bxz�t  T          @�33@L���	��@�G�BR�C��@L�����@g�B(�C��                                    Bxz�  
�          @�33@P�����@�z�BIC��q@P����Q�@XQ�B �HC��\                                    Bxz��  �          @ə�@E��Q�@��RBO��C�+�@E�����@]p�B�C��q                                    Bxz�f  T          @�33@E�@���BQp�C�n@E����@a�B33C��\                                    Bxz
  	�          @˅@<���\)@���BR�C��@<����p�@_\)BQ�C��R                                    Bxz�  
g          @�33@8Q����@�33BU�
C��\@8Q�����@c33BG�C���                                    Bxz'X  
�          @˅@>{�Q�@�
=B[�C�.@>{���@s33B�\C���                                    Bxz5�  
M          @��H@-p��3�
@��BO��C��)@-p���p�@S�
A��C�                                    BxzD�  �          @ə�@A���\@�Q�BS(�C�q�@A���
=@c33B	\)C�޸                                    BxzSJ  
�          @ȣ�@XQ��   @�z�B?=qC��@XQ����@HQ�A��
C�J=                                    Bxza�  
�          @�G�@[�����@�z�BLffC�7
@[��vff@eB�C�޸                                    Bxzp�  T          @��@^�R����@���BS{C�޸@^�R�c�
@xQ�B�RC�/\                                    Bxz<  T          @�33@X���33@�{BL��C�k�@X���}p�@fffB
p�C�O\                                    Bxz��  �          @ʏ\@E��8Q�@�
=B@�
C�l�@E����@C33A�Q�C��R                                    Bxz��  T          @��H@P  �1�@�B>(�C��q@P  ��  @C�
A�\C��                                    Bxz�.  
�          @˅@HQ��   @�{BK�RC��f@HQ����@Z=qB��C��3                                    Bxz��  �          @��
@K��{@�ffBKz�C��@K����\@\(�Bz�C�"�                                    Bxz�z  �          @ʏ\@N{�1�@�{B?G�C��H@N{��  @E�A���C��\                                    Bxz�   �          @˅@AG����@�  BPz�C�� @AG���=q@`  B33C�w
                                    Bxz��  T          @���@Mp���@�z�BV=qC���@Mp���Q�@s�
BQ�C�Y�                                    Bxz�l  T          @��
@Tz���R@�=qBR��C��f@Tz��|(�@p��B��C�                                    Bxz  �          @��@tz��#33@�G�B,=qC�'�@tz����@4z�Aՙ�C�p�                                    Bxz�  �          @ə�@S33���@�\)BC�C��
@S33���R@QG�A��C��                                    Bxz ^  
�          @��@:=q�8��@�G�BE��C��R@:=q��(�@H��A���C��                                    Bxz/  T          @ə�@?\)�:=q@��B@�C��H@?\)���H@@��A�\C��f                                    Bxz=�  T          @���@2�\�h��@���B,(�C��q@2�\��33@�A���C�Y�                                    BxzLP  �          @ə�@>{�Z=q@�(�B0p�C���@>{��@#�
A��C��=                                    BxzZ�  	�          @�G�@1녿��
@��Bn\)C���@1��h��@���B,(�C��                                    Bxzi�  T          @�Q�@@�׿�\@��B_�C��@@���p  @}p�BC�`                                     BxzxB  
�          @ʏ\@j=q���
@�{BL�
C�� @j=q�\��@vffB�HC�Z�                                    Bxz��  �          @�=q@g���@���BC��C�S3@g��u�@`  BQ�C��                                    Bxz��  
�          @ƸR@g
=���@��B5(�C��@g
=����@@  A�33C�ٚ                                    Bxz�4  
�          @ə�@XQ��%@�z�B=�RC�Ff@XQ�����@I��A�(�C�*=                                    Bxz��  "          @ȣ�@C�
�Q�@�p�BO=qC�
=@C�
��ff@`  B�HC��                                    Bxz��  
�          @Ǯ@4z��=q@�  Be��C�� @4z��u�@���B!{C�'�                                    Bxz�&  
Z          @�\)@E���G�@��B\�HC�*=@E��mp�@{�B��C��H                                    Bxz��  �          @�=q@333��\)@���BlffC��\@333�l��@�  B*�C���                                    Bxz�r  "          @��
@*=q���R@�p�Bj  C���@*=q��G�@�(�B!�
C���                                    Bxz�  �          @�33@z��+�@�Q�B`(�C�l�@z���33@l��B��C�]q                                    Bxz
�  T          @�=q@���<(�@���BS��C���@�����@X��B
=C�h�                                    Bxzd  
�          @ə�@ff�?\)@��BR�C�)@ff����@U�B =qC�"�                                    Bxz(
  �          @ȣ�?��R�s33@��B;
=C���?��R���\@'
=A�33C�<)                                    Bxz6�  
�          @�=q@��S�
@�G�BE�
C�#�@���
=@A�A噚C�H                                    BxzEV  �          @�=q@(Q��\��@��B9�RC��@(Q���Q�@0��A�{C��
                                    BxzS�  �          @��@   �}p�@�\)B(G�C�'�@   ���H@  A�\)C���                                    Bxzb�  
�          @�Q�@33���
@�33B$
=C���@33��@�A�33C�|)                                    BxzqH  	�          @�\)@(������@333A�z�C�{@(����G�?8Q�@�C�C�                                    Bxz�  
�          @�Q�@���G�@I��A�z�C��=@���p�?��A33C��                                    Bxz��  �          @�\)@33���R@7�A噚C�@33��
=?8Q�@�C�h�                                    Bxz�:  T          @�\)@33��@:�HA�Q�C��@33���R?J=q@�=qC�n                                    Bxz��  
�          @��@{��  @aG�B�C��@{���?�G�Ag�C�u�                                    Bxz��  
�          @�G�@=q����@g�B=qC�\@=q��  ?�33A~ffC�L�                                    Bxz�,  �          @�z�@   ��@Q�B{C�&f@   ���
?�(�A@��C�5�                                    Bxz��  
�          @��?�����Q�?��AQG�C���?������\�k����C��\                                    Bxz�x  
�          @���?��R�^{@b�\B&
=C��?��R���H?���A�ffC���                                    Bxz�  T          @��H?�\)��@�ffBh��C��?�\)���R@c�
B�C���                                    Bxz�  T          @�(�?�{�/\)@�=qB^�C�
=?�{��{@U�B
��C�U�                                    Bxzj  
Z          @��
?���hQ�@�Q�B;�RC�#�?������@p�AǅC�                                    Bxz!  �          @��?�
=�=p�@Z�HB4z�C�?�
=���\?���A�\)C�.                                    Bxz/�  
�          @��?�ff��\)@��\B��C�\?�ff�fff@�  B:
=C�K�                                    Bxz>\  "          @���?���{@�\)B��
C�]q?��c�
@��B8Q�C���                                    BxzM  T          @�  ?������@���B�=qC��=?���U@��HBCz�C��                                    Bxz[�  �          @�
=@�Ϳ�ff@��B|��C���@���L��@�(�B;{C�K�                                    BxzjN  �          @��?��ͿǮ@�  B�� C��
?����`  @�ffB:��C���                                    Bxzx�  T          @��\?�33�L��@�{B��C��)?�33�8Q�@�{BV�C���                                    Bxz��  �          @�p�?fff>Ǯ@�G�B���A��
?fff��
=@�\)B��\C���                                    Bxz�@  �          @�
=?�33>#�
@�33B�z�@�Q�?�33�
=q@�{B���C��                                    Bxz��  "          @�\)?�G��W
=@�B�=qC��?�G���H@�z�Bo(�C�\                                    Bxz��  
�          @���@(���G�@�33B{{C�U�@(��P��@�p�B=p�C�p�                                    Bxz�2  �          @�G�?�33�8Q�@�p�B���C��R?�33�8Q�@�B\�C��                                    Bxz��  �          @�z�@녿\@��RB}p�C�+�@��a�@�B:{C�l�                                    Bxz�~  
�          @\@�Ϳ��@���B{\)C�
@���S33@�
=B=��C�O\                                    Bxz�$  T          @���@)���	��@��B\C�t{@)���x��@k�B�C�q                                    Bxz��  �          @�=q@�����@�\)B�W
C��q@��G�@�z�BGffC��)                                    Bxzp  
�          @���@�
���@��B�ffC��@�
�L(�@�z�BL��C��R                                    Bxz  �          @�z�@(���G�@�ffBz�C��f@(��n�R@��HB4ffC�7
                                    Bxz(�  
�          @���@
�H�p�@��\BdC��{@
�H��\)@n�RB��C��                                    Bxz7b  T          @��H@��(Q�@��RBZ��C���@����H@c�
BG�C�"�                                    BxzF  �          @�33@��(�@��Bg�C��H@���Q�@z=qB 33C�J=                                    BxzT�  y          @���@'
=�@��@���BC{C�}q@'
=����@B�\A�\)C�                                      BxzcT  �          @�G�@*�H�.�R@�p�BK�C�'�@*�H��=q@QG�B{C��R                                    Bxzq�  "          @�\)@<(��,��@�{B@G�C��q@<(���ff@Dz�A�33C���                                    Bxz��  �          @��@P���E�@\)B&�C�L�@P�����
@ ��A�\)C�U�                                    Bxz�F  �          @�{@X���2�\@~�RB(��C�=q@X�����@'
=A�
=C���                                    Bxz��  T          @�p�@P���Vff@eB  C�)@P����
=@�\A��C��                                    Bxz��  
�          @�z�@>{�xQ�@P��B  C��{@>{���\?��Ar�HC�Ǯ                                    Bxz�8  
�          @��@1��c�
@Y��BC�  @1����\?��A�p�C���                                    Bxz��  /          @�z�?xQ�!G�@��RB��C�o\?xQ��*=q@��Bn(�C�                                      Bxz؄  y          @���?(���z�@��B�33C�\)?(����@��\B�{C�/\                                    Bxz�*  
�          @�{?0�׿(��@�33B�aHC�#�?0���5�@��BrG�C��                                    Bxz��  �          @�?p�׿��H@��RB�  C�ٚ?p���Q�@��\B[
=C��
                                    Bxz v  "          @�?�G��
=@�  B|
=C�
=?�G���p�@~{B)�C�g�                                    Bxz   
�          @��R?�=q�Vff@���BG�
C�T{?�=q��=q@<��A�(�C�h�                                    Bxz !�  T          @�p�@33�Tz�@�=qB;�C�T{@33��ff@0  A��C��                                    Bxz 0h  
Z          @�G�@�
�G�@�(�BH{C�J=@�
��(�@G�A���C�G�                                    Bxz ?  	`          @�33@&ff�Mp�@��B=Q�C��H@&ff���@<��A�C��
                                    Bxz M�  �          @��@z��Q�@��\B<z�C��q@z����@1�A��C�7
                                    Bxz \Z  �          @��@��w
=@n{B��C��{@���
=@ ��A�Q�C��3                                    Bxz k   "          @�@��\(�@�(�B1�HC�� @���\)@#33A��C��R                                    Bxz y�  "          @�{@C33�Q�@y��B$33C�t{@C33��  @��A��\C�3                                    Bxz �L  /          @��R@�R�fff@��B1�C��@�R��z�@!G�A�33C�B�                                    Bxz ��  y          @��@p��b�\@���B/��C�j=@p���=q@"�\A���C���                                    Bxz ��  �          @�p�@!G��tz�@n�RB��C��{@!G���@33A�Q�C��=                                    Bxz �>  �          @��@
=��  @mp�BG�C��=@
=���H?��HA�
=C�E                                    Bxz ��  y          @�{@   �HQ�@�(�BL��C�P�@   ���
@I��B {C��)                                    Bxz ъ  �          @�@G��aG�@��B:��C��
@G����@-p�A�=qC�O\                                    Bxz �0  "          @��@  �r�\@�G�B*=qC�\)@  ��Q�@�A��RC��                                    Bxz ��  "          @��R@*=q�\)@aG�B33C��3@*=q��Q�?�A�C���                                    Bxz �|  T          @�ff@  �qG�@�  B)Q�C�l�@  ��
=@A��C�.                                    Bxz!"  
�          @�{@z��W
=@�BA33C�Ф@z���  @8��A�C��f                                    Bxz!�  T          @��
@!G��U�@��HB1�\C���@!G���33@%A�ffC�Z�                                    Bxz!)n  �          @�33@0  �9��@�  B;{C���@0  ��  @9��A��C�t{                                    Bxz!8  �          @�(�@!G��dz�@y��B'
=C��)@!G����@A�  C��q                                    Bxz!F�  �          @�=q@����(�@_\)Bp�C���@�����
?�  A�  C�Q�                                    Bxz!U`  T          @��H?�����@-p�A�\)C���?���=q?W
=A�C�~�                                    Bxz!d  "          @��>�(�����?�{AxQ�C�}q>�(���녿   ���
C�j=                                    Bxz!r�  
�          @�Q�@
=q��p�@7�A��C���@
=q��p�?�AD��C���                                    Bxz!�R  y          @�=q@
�H��z�@A�BC���@
�H��ff?�=qA]��C�ٚ                                    Bxz!��  �          @��@Q�����@<(�A�G�C�AH@Q���G�?��HAG33C�s3                                    Bxz!��  y          @��@p���(�@7�A��\C�h�@p���z�?�Q�AE�C�aH                                    Bxz!�D  
�          @��@�����@.�RA�G�C���@����{?��
A)G�C�*=                                    Bxz!��  
�          @��
@&ff��  @%�Aڣ�C��{@&ff��z�?aG�A��C���                                    Bxz!ʐ  
�          @��@��@.�RA�
=C��f@���?xQ�A
=C�"�                                    Bxz!�6  �          @�=q@�����\@�RA���C�w
@����?E�A   C��3                                    Bxz!��  "          @��H@�����?�ffA��HC��@�����<��
>k�C�
                                    Bxz!��  �          @��H@=q��  ?�33A��C�t{@=q���þ\)���RC��\                                    Bxz"(  �          @�33@'���{?�=qA��C��
@'���{�8Q����C��3                                    Bxz"�  �          @��
@����Q�?ٙ�A��RC���@���������Ϳ}p�C��3                                    Bxz""t  
�          @�\)@�����@{A�\)C�{@����?5@���C���                                    Bxz"1  
�          @��R@4z���z�@�
A���C�]q@4z���p�?(�@�p�C���                                    Bxz"?�  T          @�@3�
���
?�(�A��C���@3�
�����
�#�
C���                                    Bxz"Nf  "          @�@p����?��A�{C�� @p���33=�G�?��C��f                                    Bxz"]  �          @��H?��R��ff?�{A���C���?��R��ff�aG���\C�u�                                    Bxz"k�  
�          @�ff?˅����@ ��A��C���?˅���
?E�A�\C��)                                    Bxz"zX  �          @��
?��{�@aG�B=qC��?����?�A�Q�C���                                    Bxz"��  
�          @��?�G��s�
@p��B)�C��{?�G���(�@(�A��C��q                                    Bxz"��  
�          @�Q�>��L��@�{BW�
C�AH>���G�@C33B  C�H                                    Bxz"�J  �          @�  ?�=q�W�@��BB33C���?�=q���H@+�A�{C�                                    Bxz"��  T          @���?��
�J�H@z�HB<�C���?��
���H@%�A�ffC�#�                                    Bxz"Ö  T          @��@G
=�o\)@8Q�A���C��H@G
=��  ?�z�Af�\C�Q�                                    Bxz"�<  �          @���@>�R�|(�@+�A�  C��{@>�R���
?�z�A>=qC�n                                    Bxz"��  T          @�=q@�R��(�@1�A�{C���@�R���H?�Q�AEp�C��R                                    Bxz"�  �          @��@���p�@�A�Q�C��
@����>�G�@��RC�˅                                    Bxz"�.  
g          @�{?�Q����
@0��A�ffC���?�Q�����?���A:=qC�Ff                                    Bxz#�  
          @��\@33��33@(Q�A�ffC�H�@33��Q�?��A9p�C���                                    Bxz#z  T          @�33?�\)��  ?��
A�z�C�j=?�\)��=q=L��?��C���                                    Bxz#*   
�          @�(�?�p����H@Q�A�{C��3?�p����
?+�@���C��                                    Bxz#8�  
�          @���@(��X��@aG�B �C��f@(���z�@
=qA�  C���                                    Bxz#Gl  T          @�ff?����}p�@R�\B(�C�XR?������H?�  A�G�C�L�                                    Bxz#V  �          @���?�����
@Y��B=qC�W
?������?�ffA��\C��)                                    Bxz#d�  �          @�ff@�R��p�@ffA�z�C�XR@�R��
=?J=qA�C��
                                    Bxz#s^  �          @�=q?���{@R�\B�\C�U�?����?�
=A��C���                                    Bxz#�  
�          @�=q@ ���w
=@Y��B��C���@ ������?�33A���C�h�                                    Bxz#��  �          @�ff?�G���  @P  BffC��{?�G����\?�ffA|z�C���                                    Bxz#�P  �          @���?�=q��@XQ�B�C���?�=q���?ٙ�A�Q�C�Ff                                    Bxz#��  �          @�  ?�����Q�?��
AO33C���?�����(��\)��=qC��                                     Bxz#��  �          @�33?�G���p����
�W
=C��=?�G���33�����C��\                                    Bxz#�B  
�          @�=q?�ff��=q>��R@N{C��?�ff���
��p��x(�C�ff                                    Bxz#��  
5          @��?��H�������z�C���?��H��(��\)��ffC�W
                                    Bxz#�  �          @��?�����ff��
=���C��)?����}p��:=q�
=C��                                    Bxz#�4  
          @���?0����z������\)C�}q?0���X���e�8{C���                                    Bxz$�  �          @��������G�?�z�A�\)C�Ff����������G���z�C�\)                                    Bxz$�  T          @�(�>����=q@�\A�Q�C�˅>�����?�@��C��=                                    Bxz$#&  �          @��?0������?��A\��C���?0�������\���C���                                    Bxz$1�  
�          @�z�?\(�����    ���
C��f?\(���\)�޸R���C��                                    Bxz$@r  T          @�\)?0���������C��{?0����ff���ŅC�33                                    Bxz$O  T          @�\)?��\����#�
��C���?��\��Q��\)�θRC�b�                                    Bxz$]�  0          @��?�����=p�� (�C�q�?����H������C�&f                                    Bxz$ld  x          @�?�����aG��C��H?���(������\C�
                                    Bxz${
  �          @�G�    ���?�Q�Ar�HC�      �����p��vffC�                                      Bxz$��  0          @���?�  ��(��W
=�
�HC��\?�  ���׿������C��                                    Bxz$�V  x          @��
?ٙ���{�}p��+�C�~�?ٙ���33�"�\��ffC���                                    Bxz$��  b          @��
?�����{���H�[�C�E?�����G��*=q��=qC��                                     Bxz$��  F          @��@������{�O�C��\@���e�����\C��q                                    Bxz$�H  "          @��\@Fff�c33�\(��%��C��{@Fff�E����H��33C���                                    Bxz$��  �          @�G�@Z=q�O\)�k��0(�C�<)@Z=q�1G���z����C�q�                                    Bxz$�  T          @�33@Mp��`�׿n{�1p�C�7
@Mp��A�� ����(�C�XR                                    Bxz$�:  �          @�z�@@���l�Ϳs33�4��C���@@���L����˅C���                                    Bxz$��  �          @��@H���g
=�����J=qC�~�@H���E��
�H��ffC�Ǯ                                    Bxz%�  "          @�p�@��|(��.{�C��=@��aG���33����C��                                    Bxz%,  
�          @���@:�H�e    =L��C���@:�H�Z=q����dz�C�K�                                    Bxz%*�  T          @�  ?@  ���R�Tz��%�C�?@  �}p������33C�W
                                    Bxz%9x  "          @��R�G����Ϳ�(����C����G��mp��2�\��C�{                                    Bxz%H  
�          @�=q�����  ������RC��{����c33�3�
��RC�4{                                    Bxz%V�  �          @�녿�ff���H�   ��ffCyG���ff�P  �L(��&(�Ct��                                    Bxz%ej  
�          @�����ff���H��\)�r�HC��)��ff�a��ff���C�]q                                    Bxz%t  
�          @��\@ff�[�?�Q�A�C�<)@ff�hQ�=L��?(��C�|)                                    Bxz%��  T          @�z�@0  �C�
?��HA��
C��
@0  �W�>�
=@���C��H                                    Bxz%�\  �          @���@b�\��\)@��A�p�C��3@b�\��R?�=qA�G�C���                                    Bxz%�  T          @��H@`�׾��@,��Bp�C���@`�׿�p�@(�B�
C�U�                                    Bxz%��  T          @u�@G��4z�?�  Av�\C�j=@G��?\)=�\)?��
C���                                    Bxz%�N  �          @s�
?�=q�b�\�\)�
=qC�O\?�=q�Tz῞�R����C��                                    Bxz%��  �          @~�R?}p��tzᾔz���ffC�=q?}p��b�\���H���C���                                    Bxz%ښ  T          @�
=�
=q����������\C�H��
=q�Z=q�2�\��HC���                                    Bxz%�@  �          @��þ�{���\��=q��z�C����{�l���'
=�(�C�e                                    Bxz%��  
�          @��;�����ff�������C�ٚ�����u��&ff�=qC��H                                    Bxz&�  �          @��@���}p�?�\@�
=C���@���z�H�:�H��
C��                                    Bxz&2  T          @�z�?�����H�c�
�333C��)?���w
=����z�C��                                    Bxz&#�  �          @�ff��
=��=q�����{C���
=�L���Vff�8�C�G�                                    Bxz&2~  
�          @��H�#�
��=q�	����{C���#�
�\���W��1ffC���                                    Bxz&A$  �          @��\<��
��\)�z���(�C�(�<��
�Tz��`  �:G�C�4{                                    Bxz&O�  T          @��\�L����
=�ff���HC��=�L���S33�aG��;�C��3                                    Bxz&^p  T          @�  ����|���%��ffC�����>�R�j=q�Ip�C���                                    Bxz&m  "          @�  �u��p��ff�ծC����u�U��QG��1��C��                                     Bxz&{�  T          @��>����ÿǮ��C�H>��u�7
=�{C��=                                    Bxz&�b  "          @��\?�R��G���  ���HC��q?�R�xQ��333�(�C��                                    Bxz&�  �          @�(�?B�\��(�����~�\C���?B�\��Q��*�H��HC�ff                                    Bxz&��  "          @��\?=p������  �?�C���?=p���z��
=��(�C��                                    Bxz&�T  �          @��?Tz���z�k��0  C��?Tz������G��ᙚC���                                    Bxz&��  �          @��\?��\��������HC�*=?��\���\��������C���                                    Bxz&Ӡ  "          @�33?n{��
=����
=C��?n{���\������Q�C��                                    Bxz&�F  b          @��?
=��  ���\�AG�C��\?
=��\)�����z�C���                                    Bxz&��  �          @�(�?@  ��Q�=p���
C�}q?@  ��=q�Q�����C��\                                    Bxz&��  T          @��\?�(���녾�  �>{C�|)?�(���G��˅��G�C��3                                    Bxz'8  �          @�=q?
=q������(���z�C�c�?
=q�z=q�0���C��                                    Bxz'�  �          @��H?�����\�����\C�h�?���}p��-p����C��                                    Bxz'+�  "          @�33>8Q���������Q�C�#�>8Q������*=q��
C�N                                    Bxz':*  �          @��    ������w33C�    ���H�'
=�ffC�f                                    Bxz'H�  "          @��H�#�
��{��z��^{C���#�
��z��\)��  C��\                                    Bxz'Wv  "          @�G�=�G����Ϳ���IC���=�G���z��
=��C���                                    Bxz'f  
Z          @���?k����\��\��z�C��?k��g��=p���C�&f                                    Bxz't�  
�          @��
?h����(���ff�Ip�C���?h�����
�
=���C�8R                                    Bxz'�h  �          @��?+���\)��  �<��C�f?+���\)����RC�}q                                    Bxz'�  �          @�ff?@  ���
���H��z�C�` ?@  ��Q�������C���                                    Bxz'��  "          @�Q�?W
=��{����޸RC��?W
=����=q��
=C�\                                    Bxz'�Z  T          @�ff?
=���>aG�@"�\C�p�?
=��Q쿚�H�c
=C���                                    Bxz'�   �          @�>������\?k�A+33C��{>�����z����=qC��\                                    Bxz'̦  
�          @�\)    ����?��Apz�C�      ��
=�����  C�                                      Bxz'�L  T          @�
=>k����
?p��A/�C�]q>k����   ��  C�XR                                    Bxz'��  T          @�>�  ����?��HAd(�C���>�  ����W
=��HC�xR                                    Bxz'��  �          @�p�>������?h��A,z�C�j=>����������H��=qC�b�                                    Bxz(>  
�          @�33?}p�����\)��(�C��=?}p���  ���R���
C�8R                                    Bxz(�  �          @�z�?xQ�����>k�@.{C��R?xQ���zῑ��X��C��H                                    Bxz($�  �          @��H?^�R���>L��@�C�.?^�R��33��z��_33C�U�                                    Bxz(30  
�          @�p�?E�����?
=@߮C���?E���  �J=q��C��q                                    Bxz(A�  
�          @�?��\��Q�?��@�Q�C��?��\��
=�Q��C�R                                    Bxz(P|  �          @�p�?�33��Q�>�{@|(�C�Ǯ?�33������\�@��C��                                    Bxz(_"  
�          @���?�{����?��\A<(�C���?�{���
��p���\)C�h�                                    Bxz(m�  T          @��R?�R��33�
=�ڏ\C���?�R�����
=��z�C��                                    Bxz(|n  �          @�z�?
=��=q����B�\C�z�?
=�����������C��\                                    Bxz(�  T          @�33?8Q����׾�
=��ffC�P�?8Q����R�޸R��z�C��)                                    Bxz(��  T          @��?E����׿^�R�"�HC���?E����\�
�H�ѮC��                                    Bxz(�`  
�          @�\)>��H���H�z�H�6�RC��)>��H��(��33��ffC�*=                                    Bxz(�  
�          @���>�{��{�c�
�$  C��3>�{����\)���C�%                                    Bxz(Ŭ  T          @�Q�>k�������33��ffC�c�>k���
=�+���RC��{                                    Bxz(�R  
�          @��>����R�У���Q�C��>���=q�7����C��\                                    Bxz(��  
�          @�  >�{����(���p�C��>�{�����<����
C�g�                                    Bxz(�  T          @�
=>\��\)���H��
=C�Ff>\��z��-p��z�C��R                                    Bxz) D  T          @�\)>��������p���
=C��)>�����  �<���ffC�*=                                    Bxz)�  �          @�Q�?:�H���Ϳ��H���
C�n?:�H��  �;���C�"�                                    Bxz)�  T          @���?:�H��(���ff��{C��H?:�H�}p��@����HC�AH                                    Bxz),6  T          @�G�?(���������Q�C�H?(���~�R�AG��
=C���                                    Bxz):�  T          @���>�p����
������=qC�P�>�p��z=q�HQ��\)C���                                    Bxz)I�  �          @�  >��R��=q���R���
C��3>��R�w
=�J=q�
=C�O\                                    Bxz)X(  
�          @�Q�>u�����z���G�C�y�>u�z�H�E��RC���                                    Bxz)f�  �          @�
=>�Q����׿������C�G�>�Q��u�C33�33C��                                    Bxz)ut  T          @��\�k��vff�0  �
=C�AH�k��=p��l(��G�RC|�                                    Bxz)�  �          @�ff��33�c�
�.{��RCx�쿳33�,(��e��G  Cr��                                    Bxz)��  �          @������x�������C~&f�����G
=�U��5
=Czp�                                    Bxz)�f            @�  ����e�7
=��C|W
����,(��n{�O\)Cv�R                                    Bxz)�  "          @��ÿ���w��#33�
=C}�
����B�\�_\)�<=qCyxR                                    Bxz)��  �          @�Q�=p��n{�333��C�XR�=p��5��l(��NQ�CJ=                                    Bxz)�X  x          @�  �#�
�������HC�� �#�
�U��Vff�4p�C��q                                    Bxz)��  �          @�
=?B�\���������C�?B�\�e�9���33C��\                                    Bxz)�  "          @�    ���Ϳ������C�      ��G��4z���\C�                                      Bxz)�J  T          @�(�=u��  ������{C�e=u�vff�<����C�w
                                    Bxz*�  �          @�ff�L�����ÿ�(���C���L���vff�Fff�\)C���                                    Bxz*�  �          @�{��Q���{��p���C�����Q���(��*�H�
=C�}q                                    Bxz*%<  �          @���?�����  ��z�C���?���\(��QG��,�
C��                                    Bxz*3�  �          @�\)?
=�\)�{���C�33?
=�L���[��:�C�8R                                    Bxz*B�  �          @�>��H��  �z���C�z�>��H�P  �R�\�4G�C�E                                    Bxz*Q.  �          @�>�ff�����(�����C��>�ff�j�H�QG��%�C�y�                                    Bxz*_�  �          @��
?(����z�����z�C�~�?(���W
=�[��4
=C���                                    Bxz*nz  �          @��?@  ��\)�*=q���C��q?@  �X���j�H�:=qC�0�                                    Bxz*}   
�          @�?�z���G��"�\����C��?�z��^�R�dz��.G�C�                                    Bxz*��  �          @�?(����  �(���{C�'�?(���mp��a��,�\C�f                                    Bxz*�l  "          @�33=�Q������p���p�C��=�Q��tz��Tz��#�
C���                                    Bxz*�  "          @��
>8Q����׿�=q��G�C��>8Q���z��@  ���C�Ff                                    Bxz*��  �          @��R>k���
=�*�H��Q�C�u�>k��hQ��n�R�6��C���                                    Bxz*�^  T          @���>�=q���\�S�
���C�� >�=q�E��  �W�\C�xR                                    Bxz*�  
�          @�G�>�����R�H���G�C��f>���P�����
�M�\C��3                                    Bxz*�  "          @��\>#�
�y���dz��)�RC�.>#�
�6ff��ff�e(�C��q                                    Bxz*�P  �          @�{>�{�*=q���p�HC��R>�{��\)���R�C��                                     Bxz+ �  �          @�?���� ����z��m(�C�"�?��׿��R��(��C�+�                                    Bxz+�  �          @�ff?(������~=qC�Z�?(����\������C��H                                    Bxz+B  
�          @��<#�
�	������C�
<#�
�Tz���=q¢G�C�<)                                    Bxz+,�  
�          @����  �4z�����i��C�zᾀ  ��ff��p��C�u�                                    Bxz+;�  
�          @��ͼ��HQ������W�C��q����Q����R.C��{                                    Bxz+J4  �          @��\>8Q����\�X����RC�C�>8Q��E����Y=qC��=                                    Bxz+X�  "          @��H=u��(��'���Q�C�]q=u�tz��mp��0��C�q�                                    Bxz+g�  
Z          @��\>������Q��ՅC���>���~�R�`  �%
=C�                                    Bxz+v&  �          @�33>B�\���
�U�
=C�W
>B�\�I����Q��V�C��                                     Bxz+��  
�          @�(�>k�����Z=q�\)C��q>k��HQ����\�XG�C��                                    Bxz+�r  �          @���?����z�����{C��q?���~�R�J�H�33C���                                    Bxz+�  T          @�ff?���������Q�C�o\?��������HQ��p�C���                                    Bxz+��  �          @��?��~�R�`  �$  C��?��?\)��(��\�C�l�                                    Bxz+�d  
�          @��?!G������S33�p�C�g�?!G��Fff��ff�T�C��R                                    Bxz+�
  x          @��?�p���=q���R����C��f?�p��{��C�
���C���                                    Bxz+ܰ  �          @�33?�ff��p������c33C�3?�ff��p�������p�C�h�                                    Bxz+�V  �          @�{?
=q��녿�G����C�f?
=q��G��-p�����C�`                                     Bxz+��  �          @��H��R�\)���R�T��Cd(���R�����{�yz�CRz�                                    Bxz,�  �          @���3�
�ff���T\)CX�3�
�c�
�����n��CE��                                    Bxz,H  
�          @����
=q�J=q��Q��@ffCk���
=q�G�����k�
C_\                                    Bxz,%�  
          @�  ������mp��!
=C|�{����H�����
�VffCw\)                                    Bxz,4�  �          @�=q���R���\�U���C~�{���R�hQ�����BC{{                                    Bxz,C:  �          @�(�������Q��O\)��C�P������u������<z�C}��                                    Bxz,Q�  "          @��H���H���H�
�H��(�C�` ���H�����Y���(�C��)                                    Bxz,`�  �          @��?c�
��p����
�L��C�u�?c�
����Q�����C���                                    Bxz,o,  �          @��\?�ff����>�  @(�C�xR?�ff��G������3�C��
                                    Bxz,}�  "          @�=q?�(����
>W
=@�C�O\?�(���Q쿓33�733C�t{                                    Bxz,�x  
�          @�
=?��R���\?�\@��
C�>�?��R��G��J=q���RC�N                                    Bxz,�  T          @�\)?�(����H?(��@��
C�  ?�(����H�#�
�θRC��                                    Bxz,��  �          @�\)?����
=?���A.{C���?�����\�aG��{C���                                    Bxz,�j  �          @��\?�33���?�\@��C�B�?�33���׿W
=���C�P�                                    Bxz,�  �          @��\?�����>�ff@�C�
?���
=�aG���C�+�                                    Bxz,ն  �          @��@���  ?^�RA	G�C�o\@���녾���~�RC�Q�                                    Bxz,�\  "          @�(�@,(����?�=qA*=qC��@,(���
=�#�
����C���                                    Bxz,�  	�          @�33@C�
���R?У�A�\)C��@C�
���R>�@���C��R                                    Bxz-�  
�          @��\@C33��Q�?�\)AZ{C�P�@C33��ff>L��?�p�C�Ф                                    Bxz-N  �          @���@7���Q�?�z�A��HC��\@7���Q�>�@�G�C��f                                    Bxz-�  T          @���@9�����
?�{A�Q�C�\@9����?5@��HC�=q                                    Bxz--�  
�          @�
=@.�R���
?�z�A��HC�L�@.�R��{?E�@�  C�y�                                    Bxz-<@  "          @�=q@$z���ff@G�A�G�C�XR@$z����?���A-��C�aH                                    Bxz-J�  T          @�33@'
=��33@��Aƣ�C���@'
=����?�ffAM�C���                                    Bxz-Y�  �          @��@.{��Q�@&ffA�G�C�G�@.{��  ?\Ax��C��f                                    Bxz-h2  "          @�@&ff�|��@FffB(�C���@&ff���@A��C��3                                    Bxz-v�  �          @�
=@'���?�A<��C��
@'���=q���
�8Q�C��                                    Bxz-�~  �          @���@*=q���R@��A�p�C�` @*=q��33?���A3�C�XR                                    Bxz-�$  �          @�p�@$z�����@p�A��
C�Ф@$z����?��A2�\C���                                    Bxz-��  �          @�{@*�H���
@�RA��C��{@*�H��=q?�33AdQ�C�xR                                    Bxz-�p  T          @�p�@G���33?���A^{C��@G�����>\)?�33C�y�                                    Bxz-�  T          @��@=q����?���A��
C�U�@=q��33?(��@ָRC���                                    Bxz-μ  
�          @���@ �����?���A�C���@ �����?.{@�G�C��{                                    Bxz-�b  
�          @�\)@{��  ?�z�A���C��R@{��  >�@���C�w
                                    Bxz-�  "          @�{@�����?�z�Af=qC�9�@����R>aG�@  C��{                                    Bxz-��  
�          @�Q�@p���?��Az=qC�<)@p����>�p�@mp�C��)                                    Bxz.	T  �          @�{@5��
=?�{A^ffC���@5���>�  @\)C�f                                    Bxz.�  
�          @��@�\��G�@5�A���C�g�@�\���?�(�A�p�C�.                                    Bxz.&�  
(          @��\@33��33@,��A��
C�J=@33��33?�=qA}�C�'�                                    Bxz.5F  �          @�=q@G����@(Q�A�G�C��@G���z�?�  Ao�C���                                    Bxz.C�  
�          @�33@����@��A�z�C�P�@������?��\A ��C���                                    Bxz.R�  T          @��\@�\��{@�A�33C�n@�\��G�?h��A�HC���                                    Bxz.a8  �          @�  @�\���?���A�  C�� @�\��
=?G�@��\C��
                                    Bxz.o�  �          @�
=@%���
=?�Q�A���C�S3@%���G�?Tz�A��C��                                    Bxz.~�  T          @�
=@(����@p�A�z�C���@(���G�?�{A2�RC��                                    Bxz.�*  
�          @�{@333���?�
=A�Q�C��H@333��(�?Y��A	��C��=                                    Bxz.��  �          @�p�@9������?�Ah  C��
@9����33>�{@Z=qC�n                                    Bxz.�v  �          @�ff@.{��(�?�Q�A@(�C��\@.{����=u?&ffC�7
                                    Bxz.�  "          @�  @1G����\?���A_33C��@1G�����>�=q@+�C�t{                                    Bxz.��  �          @�@E��z�?��A+�C��R@E��Q�#�
�uC���                                    Bxz.�h  �          @���@e���=q?
=q@���C��3@e����H��(����C���                                    Bxz.�  w          @��H@P����Q�>���@G
=C��@P����
=�5�陚C�\                                    Bxz.�  T          @��@{��{?�@�\)C�B�@{��{�z���  C�Ff                                    Bxz/Z  �          @��
@ff��z�?���A�  C�(�@ff��{?G�AQ�C���                                    Bxz/   �          @��R?�(���p�?aG�A�C��{?�(�����k��%C���                                    Bxz/�  �          @�p�>�33��p��˅����C��{>�33��{�0  ��ffC��                                    Bxz/.L  �          @��>�z����Ǯ��  C���>�z���ff�.{��C��3                                    Bxz/<�  �          @�(��L����z�˅���C����L������/\)���C��R                                    Bxz/K�  "          @�(����
����������
C��R���
��Q��?\)� ��C���                                    Bxz/Z>  T          @��=�G����H��\)����C���=�G������@  ���
C��=                                    Bxz/h�  "          @��>B�\������H�t(�C���>B�\����&ff��
=C��                                    Bxz/w�  T          @��H=�G���
=��{�6�\C���=�G���33�G���(�C��
                                    Bxz/�0  "          @��>\)���R��33�fffC�>\)�����#�
���
C��3                                    Bxz/��  
�          @���.{��G����R��z�C���.{��\)�E�z�C���                                    Bxz/�|  "          @��;aG������#33���HC��aG���33�e���C���                                    Bxz/�"  �          @��=u�����&ff����C�T{=u���H�hQ���C�b�                                    Bxz/��  T          @���=�\)���
�7
=��C�j==�\)��(��vff�+��C�}q                                    Bxz/�n  
�          @�<#�
����N�R�=qC�\<#�
�s33�����>�C�3                                    Bxz/�  �          @�{=L������Dz���
C�K�=L���}p������5��C�Z�                                    Bxz/�  "          @�p�>Ǯ��  �)���ޣ�C�5�>Ǯ����j=q� �
C���                                    Bxz/�`  T          @�z�>�
=����/\)��RC�w
>�
=��ff�n�R�%C��                                     Bxz0
  �          @�(�>�����33�u��+�HC�C�>����K����
�]=qC��=                                    Bxz0�  "          @�z�>�33���R�mp��%Q�C�e>�33�Tz������V�C�
=                                    Bxz0'R  T          @�(�?!G����R�@���\)C���?!G��}p��|(��233C�}q                                    Bxz05�  T          @��?�  �����Mp��
=qC�0�?�  �p������:(�C�o\                                    Bxz0D�  T          @�(�?Q���\)�\)��  C��\?Q����\�_\)�p�C�^�                                    Bxz0SD  �          @�z�?^�R������ď\C�޸?^�R��{�W
=���C��f                                    Bxz0a�  �          @��
?��\���
�z���
=C��)?��\����G
=��C�Ff                                    Bxz0p�  �          @�33?�Q����Ϳ�  ���HC���?�Q�����333��  C�*=                                    Bxz06  "          @��H?��R���Ϳ��H���HC���?��R��p��0�����C�g�                                    Bxz0��  �          @��?�����\)�7�
C��?����\�(����C�K�                                    Bxz0��  "          @��?�����(���=q�[\)C��?�����\)�Q����HC��                                    Bxz0�(  �          @��
?�G���Q쿝p��JffC�f?�G���z���
��Q�C��)                                    Bxz0��  "          @�(�?������H����=qC���?�����G��Fff��C���                                    Bxz0�t  T          @���>�{���.�R��G�C���>�{��  �mp��#��C�K�                                    Bxz0�  T          @��>L������� ����ffC�'�>L����(��`  ��C�Q�                                    Bxz0��  �          @�=q?(����\)��G�C��H?(������^{��C��                                    Bxz0�f  
�          @�z�?z���G������z�C�#�?z������6ff��=qC�u�                                    Bxz1  �          @��R?
=���\�����ffC�&f?
=��=q�<����  C�z�                                    Bxz1�  �          @�z�?=p���녾�=q�1G�C�˅?=p���(����j�RC��                                    Bxz1 X  
�          @�33?�\)����@k�B(�C�޸?�\)��ff@-p�A�Q�C���                                    Bxz1.�  �          @���?�G�����@n{B z�C��q?�G����\@1�A�(�C���                                    Bxz1=�  "          @���?�  ��@X��B�C�U�?�  ��G�@��A�(�C�G�                                    Bxz1LJ  T          @�Q�?�Q���(�@FffB�C��R?�Q���p�@�A���C��                                    Bxz1Z�  "          @�Q�?�����33@
=qA�{C��\?�����{?��A)p�C�.                                    Bxz1i�  �          @�  ?��R���H?\AuC��?��R��G�>Ǯ@xQ�C���                                    Bxz1x<  T          @��H?h�����
?�G�A ��C��)?h�����R�.{��
=C���                                    Bxz1��  �          @��\?�Q���p��#�
�uC��?�Q��������:�RC��                                    Bxz1��  "          @�  ?p����33@Z=qB
z�C�� ?p����{@A��C���                                    Bxz1�.  T          @�ff?z�H��Q�@&ffA�G�C�K�?z�H��?�(�Ad(�C���                                    Bxz1��  T          @��?G���@Q�A��C�3?G�����?�(�A<��C�Ф                                    Bxz1�z  
�          @��R?^�R��G�@#�
A�{C��?^�R���R?�A]C�XR                                    Bxz1�   
�          @�  ?�����
=@(�A���C��{?�������?��
AffC��H                                    Bxz1��  
�          @���?�p���{?�p�Ad��C�3?�p����
>�z�@1�C��                                    Bxz1�l  �          @\?�p�����?��A��C�z�?�p�����?8Q�@ڏ\C�&f                                    Bxz1�  
i          @�Q�?�z�����?�G�Ak\)C�c�?�z����R>�Q�@]p�C�                                      Bxz2
�  
�          @�\)?�=q��\)?��\A�C��=?�=q��=q�.{�˅C�q�                                    Bxz2^  
Z          @��?�  ����?�G�AAC�s3?�  ��G�=��
?L��C�G�                                    Bxz2(  
�          @��R?�����p�?E�@�{C�� ?������R�����tz�C��3                                    Bxz26�  
�          @���?�Q����H?��HA`��C�k�?�Q�����>���@8Q�C�,�                                    Bxz2EP  �          @���?�����  ?�=qA$��C�?�����33��G����
C��f                                    Bxz2S�  �          @��?�z����>�Q�@Y��C��H?�z���=q�Tz����C��\                                    Bxz2b�  
�          @���?����
>.{?˅C��?���G����\�33C��                                    Bxz2qB  
�          @���?�ff��(����
�W
=C�=q?�ff��  �����8z�C�^�                                    Bxz2�  T          @�=q?�\)��33�z����C���?�\)�����  ����C���                                    Bxz2��  �          @�Q�?�(�����@G�A��HC�#�?�(����H?xQ�A�C��3                                    Bxz2�4  
�          @�  ?���p��������C��?���\)��
=�t  C�@                                     Bxz2��  �          @���?������H?G�A ��C���?�����zᾙ���FffC��                                    Bxz2��  �          @��?������@Q�A�  C�w
?�������?�ffAMG�C��)                                    Bxz2�&  T          @�33?s33���R@;�A�G�C�ff?s33��{?��A��RC��                                    Bxz2��  T          @���?����  @�p�B,\)C��f?����  @O\)B =qC���                                    Bxz2�r  �          @��\?����[�@���B@Q�C�z�?�����{@XQ�Bp�C���                                    Bxz2�  �          @�  ?�\)��33@2�\B(�C��H?�\)���\?�A�Q�C��)                                    Bxz3�  
�          @���?У���  ?
=q@���C���?У���  ��\��G�C��H                                    Bxz3d  T          @��\?޸R��G�>L��@33C�{?޸R��
=�Y����C�0�                                    Bxz3!
  
�          @�=q?�p����>�?��C��3?�p����ÿp���33C�Ф                                    Bxz3/�  T          @���?�
=���
<#�
=��
C�O\?�
=���׿�G��)p�C�~�                                    Bxz3>V  �          @�?�\)��\)��ff��
=C�  ?�\)��G���
=�|��C�T{                                    Bxz3L�  �          @�\)@=q��33�ff��
=C���@=q��G��Mp��	Q�C�b�                                    Bxz3[�  
(          @�z�@G���{�Q��G�C�/\@G���p�������C��=                                    Bxz3jH  �          @�G�@�R��p�>��@���C�,�@�R���Ϳ(����\)C�9�                                    Bxz3x�  
�          @ə�@@����33@ffA��C�J=@@�����?�ffA=qC��                                    Bxz3��  T          @��H@2�\����@
=A��
C��
@2�\���\?��
A=qC�K�                                    Bxz3�:  "          @˅@{��=q@<(�A�Q�C�  @{��G�?��A�(�C��                                    Bxz3��  "          @�  @ �����@S33Bp�C�]q@ �����@33A��RC�K�                                    Bxz3��  �          @��H@�R���R@$z�A��
C��3@�R���
?�  A[33C��                                    Bxz3�,  
7          @�\)@p����\?У�Ar�HC�˅@p���G�?   @��C�p�                                    Bxz3��  E          @���@����?�ffAo33C��R@����>�@�p�C�^�                                    Bxz3�x  
�          @�p�@���Q�@5A�p�C��=@���
=?�{A�=qC��R                                    Bxz3�  "          @��H?����@%AӮC�W
?�����?˅A~{C��R                                    Bxz3��  "          @�  ?��H��  @A�A��C���?��H���?��RA�Q�C�4{                                    Bxz4j  T          @�{?�\��33@QG�B�C��H?�\����@�
A�C�xR                                    Bxz4  �          @�ff@S33��\)?.{@�\)C�z�@S33���׾�  �\)C�\)                                    Bxz4(�  �          @Å@Z=q�~{@>�RA�p�C�U�@Z=q��
=@	��A���C���                                    Bxz47\  	�          @�{@9���\(�@�ffBd��C��)@9����G�@�{BR��C�c�                                    Bxz4F  
�          @���@6ff��@��\Bk(�C��
@6ff�Q�@�Q�BT�
C��{                                    Bxz4T�  c          @�(�@\)��
=@��\Br�C��@\)�+�@���BU33C�q�                                    Bxz4cN  w          @�33@.{�#�
@��HB>��C�b�@.{�S33@aG�B�
C��q                                    Bxz4q�  
�          @�G�@ff?8Q�@�(�B��3A���@ff��(�@��B�.C�+�                                    Bxz4��  T          @ȣ�@ ��?!G�@���B��A�{@ �׿�@��B�L�C�                                    Bxz4�@  
Z          @�  ?��?��\@��HB��HBQ�?��=�G�@�\)B���@i��                                    Bxz4��  �          @�G�?\@��@��\B{�HBf�R?\?��
@�{B��B �                                    Bxz4��  
�          @�G�?�Q�@�R@��\B�ǮBS
=?�Q�?��@���B�\B �                                    Bxz4�2  E          @ҏ\?��
@�@���B|G�BR�?��
?�
=@��
B��{B(�                                    Bxz4��  
�          @��?�=q@N�R@���BfB�=q?�=q@
=@�G�B�33Bf�
                                    Bxz4�~  
�          @Ӆ?�\)@i��@�z�BZ�RB��?�\)@"�\@��B�8RB��                                    Bxz4�$            @�p�?���@w�@���BS(�B�\)?���@1G�@�ffB}�B��{                                    Bxz4��  �          @��?�=q@a�@�\)B]�B�?�=q@=q@���B�Bt��                                    Bxz5p  �          @�p�?L��@j=q@���B_\)B�aH?L��@!�@��
B��B���                                    Bxz5  �          @���?�  @2�\@�p�B}�RB��=?�  ?���@�33B��fBh                                      Bxz5!�  
�          @׮?���@#�
@��HB���B�k�?���?���@�
=B���BH
=                                    Bxz50b  �          @׮?�33@O\)@��RBi  B�L�?�33@@�
=B�  B`��                                    Bxz5?  �          @�
=?�p�@Q�@�Bf��B��{?�p�@Q�@ƸRB���B]G�                                    Bxz5M�  "          @�Q�?���@u�@��
BR�RB�� ?���@.�R@�  B{
=Bx
=                                    Bxz5\T  "          @�{?��\@�{@��BD\)B�Q�?��\@I��@���Bn
=B�{                                    Bxz5j�  �          @ٙ�?�ff@,��@��RBuBa
=?�ff?�G�@��
B�z�B�R                                    Bxz5y�  
�          @��?�@Fff@�
=Bk�BmQ�?�?�33@�ffB�ffB7�                                    Bxz5�F  �          @�z�?��
��z�@{�Bi(�C�z�?��
�Q�@b�\BH(�C�^�                                    Bxz5��  
�          @�ff@�R�>{@g�B,Q�C���@�R�g
=@?\)B	33C�E                                    Bxz5��  �          @���@���Q�@���B[(�C��
@��/\)@|��B<�RC��                                    Bxz5�8  
�          @�G�@#�
>W
=@��
Bx��@��@#�
�:�H@�=qBtp�C�{                                    Bxz5��  
�          @�p�@N�R?��@�Q�BV�
A��\@N�R>��@�BbG�@�                                    Bxz5ф  T          @���@B�\@��@�\)BNQ�B�
@B�\?�G�@��Bd  A�z�                                    Bxz5�*  
�          @ƸR@)��@P��@���B<�BK��@)��@@�=qB\�B%�R                                    Bxz5��  �          @��@	��@�z�@Z�HB�B��@	��@j=q@�{B2��Bn33                                    Bxz5�v  
�          @�z�?���@��R@N{A��B��H?���@�Q�@��B(z�B��                                    Bxz6  �          @�{?ٙ�@�(�@W�B�B�� ?ٙ�@���@��B-�\B�\)                                    Bxz6�  T          @���@G�@��@.{A�
=B���@G�@�z�@dz�B�\Bt�                                    Bxz6)h  T          @�33@�@��\@5�A�  B�\)@�@}p�@i��B=qBwp�                                    Bxz68  �          @��@@�z�@e�B�B|�H@@X��@���B<��BiG�                                    Bxz6F�  �          @��R@�@7�@�
=BO
=BZ��@�?�(�@�Bpz�B0z�                                    Bxz6UZ  �          @�  ?�Q���@}p�B�p�C���?�Q쿔z�@s33B�(�C���                                    Bxz6d   �          @�z�?�=q�c�
@�z�B�  C�K�?�=q���H@�z�Br�C�h�                                    Bxz6r�  	o          @�?�(��=p�@�B�{C��H?�(�����@�ffB{�C��q                                    Bxz6�L  
          @�{?�?�(�@��B��fB�?�>���@�z�B�A#�                                    Bxz6��  �          @�Q�@
�H?L��@��\B|��A�=q@
�H���
@���B��
C���                                    Bxz6��  �          @���@L��@
=@���B=�B�
@L��?��@�
=BQ��A��                                    Bxz6�>  �          @��@`  @ ��@~{B*��Bz�@`  ?��H@�(�BA(�A�ff                                    Bxz6��  �          @��@aG�@3�
@n{B  B�@aG�@�
@�B6�
A�\                                    Bxz6ʊ  T          @�
=@K�����@u�B?\)C�B�@K���@`  B*�C�e                                    Bxz6�0  �          @�=q@ff@7
=@S33B&�BJ33@ff@�@q�BF��B+ff                                    Bxz6��  �          @���?��R@�p�@7
=A�\B�\?��R@�G�@o\)B\)B�W
                                    Bxz6�|  �          @�Q�?�z�@��@$z�AɮB�{?�z�@���@_\)BffB�aH                                    Bxz7"  �          @��R@��@��R?��A��B�ff@��@�Q�@1�A�
=B�\)                                    Bxz7�  �          @��@�@�\)?�\)A��B��
@�@��H@!�Aљ�B|=q                                    Bxz7"n  �          @���@��@��?��RAB{B��@��@��@p�A�  B�{                                    Bxz71  �          @��@/\)@��H?@  @��HB{33@/\)@�33?޸RA�  Bv�                                    Bxz7?�  �          @���@'
=@�G�?��
ADz�B~��@'
=@�ff@��A��Bx�
                                    Bxz7N`  �          @��H@��@�ff?5@��HB��\@��@�
=?�
=A�G�B{                                    Bxz7]  �          @���?��R@��H?(�@˅B�=q?��R@�(�?ǮA��
B���                                    Bxz7k�  �          @�ff@  @$z�@b�\B8  BCp�@  ?�{@}p�BV�\BQ�                                    Bxz7zR  �          @��R?�>���@���B�u�AX(�?���@�z�B���C��                                    Bxz7��  �          @��?��ͼ��
@�z�B���C�|)?��Ϳfff@���B���C��                                    Bxz7��  �          @�Q�?��y��?^�RA9��C��{?��\)=�?ǮC�U�                                    Bxz7�D  �          @��?O\)���ÿ�z��c
=C�3?O\)��\)�   ��33C�l�                                    Bxz7��  �          @�G�?�G���z����  C�"�?�G���
=�������C�]q                                    Bxz7Ð  �          @���?޸R�~{?�(�Az�HC��
?޸R��(�>�G�@��C�n                                    Bxz7�6  �          @�33?��H�g
=?ٙ�A�(�C���?��H�w�?}p�AQC���                                    Bxz7��  �          @�Q�?�p��c�
?��
Aƣ�C�N?�p��u�?�=qAk\)C��3                                    Bxz7�  �          @�G�?����>8Q�@(�C���?����
�(��C��                                    Bxz7�(  �          @��?���n{?�33A�C��{?���z�H?.{A\)C�t{                                    Bxz8�  �          @HQ�=�?�  ?�G�B4p�B��)=�?��@�
B_�HB��q                                    Bxz8t  �          @��=L��@w
=@�
A�p�B�u�=L��@Vff@@  B'Q�B�=q                                    Bxz8*  T          @�=q=#�
@xQ�@Q�A�B���=#�
@W
=@Dz�B)z�B���                                    Bxz88�  �          @��H����@���@	��A�{B��ý���@c33@7�B��B�\)                                    Bxz8Gf  �          @�(�>�p�@Z�H@*=qB�HB�� >�p�@6ff@P��BB33B�\                                    Bxz8V  �          @�(�?G�@Z�H@%�B{B�p�?G�@7�@K�B;p�B�                                    Bxz8d�  �          @���?0��@Dz�@%�B33B��R?0��@!�@G
=BG��B��                                    Bxz8sX  �          @dz�>#�
@-p�@�\B z�B�#�>#�
@�R@0��BL
=B���                                    Bxz8��  �          @\�;�p�@1G�?�33B�HB�8R��p�@
=@��B4{B��
                                    Bxz8��  �          @L�Ϳn{@1�?�
=A��B��
�n{@ ��?�Q�B ��B܏\                                    Bxz8�J  �          ?�ff�Ǯ?��
?\(�A�z�B�ff�Ǯ?��?���BffB�=q                                    Bxz8��  �          @;���  @(�?��\AͅB�����  @
=q?��HB�
B���                                    Bxz8��  �          @J�H�c�
@*�H?�33A�p�B��f�c�
@
=?��B��B�k�                                    Bxz8�<  �          @u���R@HQ�?�\)A�=qB��H���R@1G�@�B33B�{                                    Bxz8��  �          @��H��\)@g
=?�G�A�G�B�B���\)@Mp�@=qB33B�u�                                    Bxz8�  �          @�=q�Q�@^�R?���A�ffB����Q�@H��@�A�{B�#�                                    Bxz8�.  �          @�녿�\@j=q?�33A�B�׿�\@U�@z�A��B�
=                                    Bxz9�  �          @�Q��(�@l(�?�
=A���B����(�@Y��?�{A�ffB���                                    Bxz9z  �          @��Ϳ�Q�@}p�@A���B�#׿�Q�@_\)@333BG�B�\                                    Bxz9#   
�          @�33��33@xQ�@-p�B ffB�B���33@R�\@Y��B'�B�
=                                    Bxz91�  �          @�{�G�@e@K�B�\B��G�@:�H@s33B;�
B�aH                                    Bxz9@l  �          @�����R@c�
@333B	B� ���R@=p�@[�B/33B��H                                    Bxz9O  �          @�\)���@J=q@EB{B�#����@ ��@h��B<ffC��                                    Bxz9]�  T          @����R@7
=@R�\B(��B��R��R@(�@qG�BI�C�=                                    Bxz9l^  �          @�Q���@\(�@J�HB\)B�ff��@1G�@p��B8p�C#�                                    Bxz9{  �          @�Q��
=@g
=@<��B	ffB�ff�
=@>�R@eB-�C c�                                    Bxz9��  �          @�����
@8Q�@4z�B\)B�33��
@�\@Tz�B<�RC                                    Bxz9�P  �          @�\)�G�@*=q@?\)B'\)B�G��G�@�\@\(�BHC�H                                    Bxz9��  
�          @�p��,(�@>�R@J�HB33C��,(�@z�@k�B8{CB�                                    Bxz9��  �          @����G�@e�@l(�B.=qB�Ǯ��G�@3�
@��BV
=B왚                                    Bxz9�B  �          @�녿��@N�R@dz�B3��B����@�R@��
BZB�                                     Bxz9��  �          @�  ��\)@\(�@k�B3p�B�aH��\)@*�H@���B[�B�=q                                    Bxz9�  �          @�=q����@S33@\)BC�HB��H����@{@���Bm{B���                                    Bxz9�4  �          @����z�H@X��@z�HB@�B�G��z�H@$z�@�  Bj
=Bݳ3                                    Bxz9��  �          @�zῷ
=@J�H@Q�B-G�B���
=@�R@tz�BT�RB���                                    Bxz:�  �          @�33��\)@HQ�@"�\B�HB�R��\)@%@EB5\)B��                                    Bxz:&  �          @��ÿ�Q�@U�@.{B�B�R��Q�@/\)@S�
B;�B�G�                                    Bxz:*�  �          @�33��z�@l(�@(��B�B�G���z�@G�@S33B,\)B��                                    Bxz:9r  �          @��Ϳ��R@u�@(��B��B�aH���R@P��@U�B+�B��                                    Bxz:H  �          @��R����@X��@>{B�\Bߔ{����@0  @c�
BE�B�aH                                    Bxz:V�  �          @�  ���
@Vff@9��BB�(����
@.{@_\)B<  B�u�                                    Bxz:ed  �          @��\��@�R@VffB=Q�B�#׿�?��@qG�B^�HC��                                    Bxz:t
  �          @�z῵@:=q@Tz�B7  B�𤿵@{@tz�B]�HB�=q                                    Bxz:��  �          @�\)����@J=q@=p�B%�RB�LͿ���@!�@`��BO33B��H                                    Bxz:�V  �          @��R�E�@G�@EB/��BϽq�E�@p�@h��BZ�RBֽq                                    Bxz:��  �          @���G�@I��@P��B4p�B��)�G�@p�@s�
B_ffB�=q                                    Bxz:��  �          @����^�R@J�H@H��B.��Bҽq�^�R@   @l(�BY=qB�aH                                    Bxz:�H  �          @�{��(�@<��@Z�HB<{B�#׿�(�@�R@{�BdffB홚                                    Bxz:��  �          @�  ���H@Fff@@��B(z�B޸R���H@p�@c33BQffB�                                    Bxz:ڔ  �          @�G����@R�\@+�B��B��Ϳ��@,��@P��BF��B�(�                                    Bxz:�:  �          @�G���@W
=@%B�HB��f��@333@L(�BA  B�u�                                    Bxz:��  �          @�����=q@S�
@!G�BffB�  ��=q@0��@G
=B9��Bޙ�                                    Bxz;�  �          @��(��@R�\@333B�Bʔ{�(��@,(�@XQ�BJ�Bπ                                     Bxz;,  �          @�
=�&ff@J�H@C33B-33B�8R�&ff@!G�@g
=BX��B�                                    Bxz;#�  �          @�G��z�@p  @��A�33Bţ׿z�@P  @8��B$�B�L�                                    Bxz;2x  �          @�=q�c�
@p��@�B �\Bγ3�c�
@Mp�@G
=B,�B�                                    Bxz;A  �          @�(����@~{?�Q�A��RBҞ����@g
=@(�A�G�BՀ                                     Bxz;O�  �          @��\���@|(�?���A�
=B��f���@g
=@z�A�Bՙ�                                    Bxz;^j  �          @�{��G�@�G�?�(�A�G�B��=��G�@g�@{Bz�B���                                    Bxz;m  �          @�  ��=q@n�R@��B\)B�ff��=q@L(�@Dz�B/ffB���                                    Bxz;{�  �          @�z�=��
@�33?�z�A�Q�B���=��
@�{@!�A��RB�Ǯ                                    Bxz;�\  �          @�  ��Q�@���?�\A�G�B�p���Q�@~{@&ffB�HB��3                                    Bxz;�  �          @�{��@���?�A��B�uþ�@s�
@.{B�B��H                                    Bxz;��  �          @��=��
@�Q�?\A��B�=��
@xQ�@�A�B��=                                    Bxz;�N  �          @�33?�@���?���A��B�Q�?�@n{@&ffB
�B��                                     Bxz;��  �          @���?��H@�33�W
=�!G�B���?��H@���?333A�B�W
                                    Bxz;Ӛ  �          @��H?�  @��H��z��n{B��)?�  @��?�@�RB��                                    Bxz;�@  T          @��?���@�
=?�
=A��\B�B�?���@s33@�RA��B��R                                    Bxz;��  �          @��?Tz�@�Q�?�Q�A�Q�B���?Tz�@u@   B�
B�u�                                    Bxz;��  �          @���?�z�@tzἣ�
���RB�=q?�z�@p  ?=p�A)B���                                    Bxz<2  �          @��?�
=@.�R�N�R�=(�B�(�?�
=@Tz��'����B��R                                    Bxz<�  �          @��?�\@l(���ff�f�HB���?�\@tzᾙ����=qB�33                                    Bxz<+~  �          @�(�@�
@c�
?Y��A5G�Bd(�@�
@Tz�?��A���B\                                    Bxz<:$  �          @��H@Q�@o\)��ff���HBqz�@Q�@~{�G����Bw\)                                    Bxz<H�  �          @�?�p�@|(���=q���RB�p�?�p�@�
=��G��H(�B�W
                                    Bxz<Wp  �          @���?��H@n�R����ׅBx�?��H@��\��{��33B���                                    Bxz<f  �          @�\)?���@s33�\��=qB�  ?���@��ÿ:�H�B��{                                    Bxz<t�  �          @�z�?���@��ͿG��
=B�
=?���@�
==�\)?^�RB�                                    Bxz<�b  �          @��\?�p�@�녿8Q��G�B�H?�p�@��
=�G�?���B���                                    Bxz<�  �          @�=q?��@���(���z�B�Ǯ?��@���>k�@8��B�.                                    Bxz<��  �          @�  ?�\)@fff��=q��
=B��?�\)@x�ÿ����lz�B���                                    Bxz<�T  �          @��
?aG�@Fff� ����B�33?aG�@b�\��=q��\)B�                                      Bxz<��  �          @�=q?�  @j�H����33B�G�?�  @~{�����e�B��                                    Bxz<̠  �          @�33?޸R@u���
����B��?޸R@��
�xQ��C�B�{                                    Bxz<�F  �          @�?��@hQ��(���RB��{?��@�  ��z����HB�k�                                    Bxz<��  �          @�{?h��@����z��ՅB��\?h��@����(��t��B�u�                                    Bxz<��  �          @�{?n{@������d  B�G�?n{@����k��3�
B�                                      Bxz=8  �          @�p�?���@�=q����Yp�B���?���@�ff�B�\���B��f                                    Bxz=�  �          @��R?
=q@�p���z���Q�B�{?
=q@��
����p�B���                                    Bxz=$�  �          @���?�
=@hQ����� G�B�{?�
=@�G���������B��                                    Bxz=3*  �          @�ff?xQ�@��׿������\B�k�?xQ�@�
=�����33B��\                                    Bxz=A�  �          @�z�?�33@��H�c�
�'�B��?�33@�=�\)?J=qB��                                    Bxz=Pv  T          @�z�?��H@�ff�\)����B�=q?��H@�
=>�
=@�(�B�Q�                                    Bxz=_  �          @���?�p�@�zᾸQ����B�?�p�@��
?��@�  B��{                                    Bxz=m�  �          @���?n{@���+����B��
?n{@���>��
@o\)B�                                      Bxz=|h  �          @�z�?�  @�ff�c�
�'�B���?�  @���=�Q�?�{B�\)                                    Bxz=�  �          @��R?h��@��H�����HB���?h��@��\?�@��B�Ǯ                                    Bxz=��  �          @��R?��H@�����Q쿈��B���?��H@�
=?fffA&�HB��                                    Bxz=�Z  �          @�z�?���@��R����{B�z�?���@�z�?\(�A"�\B�                                    Bxz=�   �          @�Q�=�G�@��>�Q�@�G�B�B�=�G�@�=q?��\A��B�#�                                    Bxz=Ŧ  �          @��;��@�33?!G�A�HB��H���@�(�?�G�A���B�{                                    Bxz=�L  �          @�=q>��@���>��@�(�B�p�>��@��H?�=qA�ffB�.                                    Bxz=��  �          @���#�
@�
=>��@K�B��ý#�
@��?�p�Ar=qB���                                    Bxz=�  �          @��R<��
@���?333A	B�z�<��
@���?�33A��RB�p�                                    Bxz> >  T          @��<��
@�
=?fffA,z�B���<��
@�?�{A�=qB���                                    Bxz>�  �          @�ff�#�
@�G�?�As�
B�ff�#�
@|��@�A�{B�k�                                    Bxz>�  �          @��\�B�\@vff?��
A�
=B��\�B�\@`  @�
A�G�B��                                    Bxz>,0  �          @�z�u@��R?��HAq�B�uýu@��@
=AمB��{                                    Bxz>:�  �          @�
=��Q�@�ff?�G�A�\)B�B���Q�@�G�@=qA��B�33                                    Bxz>I|  �          @�녾k�@��R?p��A$(�B���k�@�z�@ ��A���B�L�                                    Bxz>X"  �          @�z�k�@��R?���Ag�B�
=�k�@��@=qAԸRB�u�                                    Bxz>f�  �          @�{��G�@�p�?�\)A��\B��׾�G�@�
=@*�HA�G�B��\                                    Bxz>un  �          @��H�
=@�G�@�AŅB�aH�
=@�{@UB
=B�.                                    Bxz>�  �          @��׿L��@��\@&ffA�Q�B��H�L��@�p�@g
=B33BȀ                                     Bxz>��            @��׿��
@��@6ffA�ffB˽q���
@�ff@tz�B%�
Bϣ�                                    Bxz>�`  �          @���Ǯ@�p�@�A�z�B�� �Ǯ@��
@C�
B��B�Ǯ                                    Bxz>�  �          @�G��u@�(�?�p�A���B�ff�u@��@:�HBz�B��{                                    Bxz>��  
�          @��H�.{@�Q�@z�A��
B�Q�.{@z=q@N�RB��B���                                    Bxz>�R  �          @��Ϳ��@��
@*=qA��\B�Q���@mp�@a�B-  B��)                                    Bxz>��  �          @�ff���H@��R@(Q�A�G�B�� ���H@r�\@aG�B*�\B®                                    Bxz>�  �          @�(���@|��?�=qAƏ\B���@^{@(Q�B��B��)                                    Bxz>�D  T          @`��?��@�E���B~  ?��@p��������HB�p�                                    Bxz?�  �          @��H@(��<������L��C�C�@(���z����H�qG�C�w
                                    Bxz?�  �          @�ff@  �*�H����H  C�
=@  ���H��(��jz�C�S3                                    Bxz?%6  �          @�
=@�������G��P�RC�W
@�׿�(���
=�q�C�|)                                    Bxz?3�  �          @��@��p������QQ�C�Ff@���G������n�
C��                                    Bxz?B�  �          @�@�{�n{�D
=C�B�@��\)����b=qC�޸                                    Bxz?Q(  �          @�
=@G���H�l���@G�C��)@G��Ǯ��(��a{C��H                                    Bxz?_�  
�          @��@���hQ��>�C���@��p������^ffC���                                    Bxz?nt  T          @�p�@���8���J=q�#�C���@������mp��I��C��=                                    Bxz?}  �          @���?�
=�o\)�1��
��C�z�?�
=�C33�aG��9�C��
                                    Bxz?��  �          @��H?�����{������HC�  ?����g��@  ���C�+�                                    Bxz?�f  �          @�(�?��H���R�
�H����C��?��H�hQ��A����C�>�                                    Bxz?�  �          @���?�ff��Q������G�C���?�ff�XQ��QG��(\)C��                                    Bxz?��  �          @���?\(�����'
=��C�q�?\(��s�
�a��(z�C�c�                                    Bxz?�X  �          @�z�?!G������=p���33C�Ǯ?!G���  �|(��0��C�}q                                    Bxz?��  �          @�z�?5����\(��z�C���?5�b�\���\�H�\C���                                    Bxz?�  �          @�?h���Y���s33�<�\C���?h����R��ff�m(�C�                                      Bxz?�J  �          @���?+��AG��z=q�N��C�9�?+�����\)���C��                                     Bxz@ �  �          @��H?=p��h���s�
�6��C��f?=p��-p������hp�C��                                    Bxz@�  �          @�{?G��qG��u��3��C�� ?G��4z���=q�ep�C�                                    Bxz@<  �          @�Q�?���(��Q����C�c�?��b�\��p��E�C�/\                                    Bxz@,�  
�          @�(�?�33�l(���Q��K�C��3?�33�!����R�{�HC�@                                     Bxz@;�  �          @�ff>�=q����p���
=C��>�=q���a��  C��                                     Bxz@J.  �          @��?Q������AG���z�C��3?Q���  �����2
=C���                                    Bxz@X�  �          @���?
=��Q��dz����C��
?
=�e����K�HC��f                                    Bxz@gz  �          @�(�?!G������~{�)C�0�?!G��S33���\�\�C�p�                                    Bxz@v   �          @���?\(������=q�7Q�C��=?\(��A���(��i�
C���                                    Bxz@��  �          @�?fff�e��(��M\)C�  ?fff�p���=q�G�C��                                    Bxz@�l  �          @�\)?z�H�h����(��Kp�C�~�?z�H�   ���H�}(�C��                                    Bxz@�  �          @�G�?�33�mp���33�GQ�C��?�33�$z���=q�x\)C�
                                    Bxz@��  �          @\?�  �p������EG�C�:�?�  �'����H�u��C�Ф                                    Bxz@�^  �          @�=q?�33�tz���\)�?(�C��?�33�,������o\)C���                                    Bxz@�  �          @\?ٙ��z=q���\�5�C���?ٙ��4z�����d�
C���                                    Bxz@ܪ  �          @�(�?��������s�
�!G�C��q?����S33���R��C�q                                    Bxz@�P  �          @�33?Tz����H�r�\�"ffC�g�?Tz��W
=���V{C��                                    Bxz@��  �          @���?��
����X���	33C��q?��
�}p�����<C�O\                                    BxzA�  �          @�p�?B�\���H�9����\)C�@ ?B�\��G��\)�*
=C��                                    BxzAB  �          @�(�?333�����
����C��H?333��G��L(��
��C�e                                    BxzA%�  �          @��H>�Q���\)�@  ����C��>�Q��������0�C�t{                                    BxzA4�  �          @�ff?+����H�C33���C�� ?+���  ��z��/��C�z�                                    BxzAC4  �          @���?0�����\�Mp�� G�C��q?0����ff�����533C��=                                    BxzAQ�  �          @��?�\)����fff�G�C��H?�\)�x����(��C33C��                                    BxzA`�  �          @�ff?��\���\�j=q�C�]q?��\�vff���DQ�C�!H                                    BxzAo&  �          @��
?�=q���n{���C���?�=q�k����R�I  C��                                    BxzA}�  �          @���?��H�����`  �p�C��?��H�u���Q��AffC��                                    BxzA�r  �          @���?}p��U���p��I  C�K�?}p�������H�{�C�ٚ                                    BxzA�  �          @��H?���������\)L�C�Ф?����B�\���C���                                    BxzA��  �          @��?L�Ϳ�����\)� C�T{?L�;\���R ��C�9�                                    BxzA�d  �          @�=q?c�
��z����.C��f?c�
�5��p�.C���                                    BxzA�
  �          @�ff?��H�B�\��p�C���?��H?�����\33B"��                                    BxzAհ  �          @�(�?n{?�
=����
Bt{?n{@3�
��ff�aQ�B�\)                                    BxzA�V  �          @�(�?��H?������v��B<
=?��H@8Q����\�Jp�BmQ�                                    BxzA��  �          @�(�?��R?����z�=qB?�?��R@0������Y33Bv�R                                    BxzB�  �          @�?#�
?�=q���{B���?#�
@0  ���H�i�B���                                    BxzBH  �          @���>\?�
=����fB���>\@Dz���z��[G�B���                                    BxzB�  �          @��\��@5�`���Jp�B�aH��@e�.�R�B�\)                                    BxzB-�  �          @�(�>��?�33������B���>��@!���p��o�B�aH                                    BxzB<:  �          @�?Q�?�����{B^�H?Q�@ff��  �t�B��=                                    BxzBJ�  �          @�=u?������R� B���=u@   ����s\)B�33                                    BxzBY�  �          @��R?�R>�����£L�B�R?�R?�\)��=qaHB�8R                                    BxzBh,  �          @��?�G���  ���
{C��{?�G�?}p�������B2=q                                    BxzBv�  �          @�=q?�����aHC��?�?������.B0�R                                    BxzB�x  �          @���?�Q�>u��G��
A8Q�?�Q�?��H���\\)BJ�H                                    BxzB�  �          @�ff?u�n{��  .C��?u>��
���\�{A���                                    BxzB��  �          @�Q�>\�����{¦��C�~�>\?O\)���� W
B��                                    BxzB�j  �          @�(�?G��Y�����HC�:�?G�>�G���\) ��A�p�                                    BxzB�  �          @�33?u��
=��\)aHC�5�?u?p�����8RB1=q                                    BxzBζ  �          @����Q��Vff�(���  C�E��Q��&ff�XQ��Q��C�\                                    BxzB�\  �          @��\>W
=���\�\)��{C��>W
=���׿�z���G�C�!H                                    BxzB�  �          @���?xQ���=q����ffC�+�?xQ����\�dz��=qC���                                    BxzB��  T          @��R?��
��p��������\C�� ?��
�����N{�{C��                                    BxzC	N  �          @�Q�@����{��Q���\)C�@����G��J=q��p�C�4{                                    BxzC�  T          @�@(������
=q��C�\)@(����(��Q��ffC�J=                                    BxzC&�  �          @�{@)�������\���Q�C���@)���C33��33�<\)C��                                    BxzC5@  �          @��@�
��=q�c33�33C��q@�
�S33��Q��D��C���                                    BxzCC�  �          @�  @���{��z�H�#\)C��\@���5������Sp�C�L�                                    BxzCR�  �          @��R?�Q���Q������+�\C�ff?�Q��7���p��_��C�5�                                    BxzCa2  �          @�?��H�e��  �G�RC�J=?��H�ff��Q��}z�C��q                                    BxzCo�  �          @��?�
=�����{��'��C�W
?�
=�:=q��=q�\Q�C�                                    BxzC~~  T          @�z�?����p��Z�H���C���?���Z�H���F��C�!H                                    BxzC�$  �          @��>k��O\)����`{C�>k�����\).C�u�                                    BxzC��  �          @�33>\�L����33�a�\C�j=>\��\)��  ǮC�˅                                    BxzC�p  �          @��
>L���8Q����H�q�C��R>L�Ϳ�  ����C��f                                    BxzC�  �          @�z�<#�
�Tz���=q�]��C��<#�
���R��  L�C�q                                    BxzCǼ  �          @����R�Z�H���R�VG�C��
��R�
=���C}�                                    BxzC�b  �          @�=q=u�mp����H{C�z�=u�p���\)��C���                                    BxzC�  �          @���?��
�4z���ff�j��C�f?��
���H��Q�
=C��
                                    BxzC�  �          @�=q?�z��S33���\�R�C��\?�z�� ������p�C��R                                    BxzDT  �          @�Q�?#�
���H�s33�ffC��q?#�
�]p���33�XG�C�E                                    BxzD�  �          @Å=��
�}p���z��F  C���=��
�(Q���Q��)C��)                                    BxzD�  �          @�z�>����z������>(�C�H>���5���{�z
=C�xR                                    BxzD.F  �          @��H?c�
�J�H����d��C��q?c�
�޸R��  k�C��{                                    BxzD<�  �          @�G�?��\�;����\�g�C��H?��\��G���p���C��                                    BxzDK�  �          @�33?����'���=q�s��C��q?��׿�33��=qǮC�!H                                    BxzDZ8  �          @��?У׿�������ǮC�)?У�=�\)���\��@\)                                    BxzDh�  �          @��?�{�˅��G��{C�� ?�{��Q�������C�k�                                    BxzDw�  �          @��@�Ϳk����R��C�� @��?
=q��Q�Q�A\��                                    BxzD�*  �          @��@�>������L�A��@�?�(����\�v�HB\)                                    BxzD��  T          @�p�@��>�Q����\�
A
{@��?�Q������j33B\)                                    BxzD�v  �          @��R?�>���
=��Al  ?�?�������x33B4\)                                    BxzD�  �          @�Q�?Y����33�����C�U�?Y��>�Q����¡aHA�(�                                    BxzD��  �          @�
=?�p��Ǯ�����
C��?�p���������ǮC��q                                    BxzD�h  �          @�@p�����p��C�W
@p�?^�R��(�8RA�p�                                    BxzD�  �          @��
@Q�n{���H�C�0�@Q�>����33ABff                                    BxzD�  �          @��
@�׾�ff���H8RC�aH@��?k������A��
                                    BxzD�Z  �          @��
?�׿c�
��ff�C�B�?��?(����\)�3A�                                    BxzE
   �          @�  ?J=q�+������x�C�7
?J=q��z����C�{                                    BxzE�  �          @���?������
��p��qC��?��������ff��C��=                                    BxzE'L  �          @��?�z�������wC��q?�z�^�R�����qC�<)                                    BxzE5�  �          @��
?�Q��Dz���p��T�C�h�?�Q��Q����\�3C�w
                                    BxzED�  �          @��
?��H�vff����/��C��3?��H�'
=��ff�g{C��
                                    BxzES>  �          @�=q?�ff�}p���Q��/z�C��?�ff�.�R���j{C��{                                    BxzEa�  �          @�������C�
��Q��jG�C�
=���Ϳ�=q�����C�&f                                    BxzEp�  �          @�{�+��R�\��(��]�HC�9��+�����33z�Cy�H                                    BxzE0  �          @��\����X����ff�WQ�C�ff��Ϳ�p���ff�C~z�                                    BxzE��  �          @�G�>k��A���p��iQ�C�&f>k��Ǯ���B�C�%                                    BxzE�|  �          @��R>k��2�\��
=�r�C�\)>k����������C��
                                    BxzE�"  �          @���?B�\��  ���33C��\?B�\�B�\��{¤=qC��)                                    BxzE��  �          @��H?\(��{����3C���?\(��+���
=ffC��                                    BxzE�n  �          @�?\)�E�����g�C�+�?\)��=q�����HC�Ǯ                                    BxzE�  �          @�Q�?}p��2�\���R�q�C��H?}p����R����L�C�K�                                    BxzE�  
�          @�  ?G��(�����\�z�RC�8R?G������33�3C�0�                                    BxzE�`  
�          @�=q?\)�/\)���H�y\)C���?\)��33��z��C��                                    BxzF  �          @�33?\)�C33��
=�m  C�8R?\)��p������C�n                                    BxzF�  �          @�G�?��Vff��
=�^p�C��)?녿�=q���R�)C���                                    BxzF R  �          @\>��R�O\)���
�fG�C���>��R��
=���\ffC�1�                                    BxzF.�  �          @��H>#�
�]p���  �\��C�W
>#�
������
=C�j=                                    BxzF=�  �          @��>�ff�%��
=�RC��{>�ff�^�R��
=¢{C��
                                    BxzFLD  �          @��
��  �Mp���\)�nG�C�� ��  ���
��p��3C�P�                                    BxzFZ�  �          @�33���G
=����p��C�1����
=����8RCy�                                     BxzFi�  �          @�33�!G��"�\��G�aHC�ÿ!G��J=q�ȣ�¡��CgO\                                    BxzFx6  �          @��þ�G���������C����G��8Q��Ǯ«\)CJ��                                    BxzF��  �          @�(�>L������Q���C���>L�;��
�˅­.C�:�                                    BxzF��  �          @��H>��
��{����\)C���>��
�u�ʏ\®33C��{                                    BxzF�(  �          @��H?0����
���RC�8R?0�׾�\)����¦� C���                                    BxzF��  �          @�33?�(��(���
=��C�@ ?�(��333��aHC��                                    BxzF�t  �          @��H?��
=���#�C���?���{��
=C��                                    BxzF�  �          @�?�z������H\C��{?�z�=�G����Q�@�                                    BxzF��  �          @���?�{��p�����C�k�?�{>�=q�����\A\��                                    BxzF�f  �          @�  ?B�\�B�\��z� 33C�p�?B�\?�ff��33
=BX�                                    BxzF�  �          @��H>u��z���=q­=qC�޸>u?Ǯ��z�(�B��{                                    BxzG
�  �          @ʏ\��>aG��ə�¯\)B�k���@�\��
=�B���                                    BxzGX  �          @�=q���\��G�­�C��=��?�(���(�  B�#�                                    BxzG'�  �          @�녾�  >���G�®�HC����  ?�(���\)W
B�aH                                    BxzG6�  �          @��
=����(���{�y��C�\)=��Ϳ^�R�����qC�j=                                    BxzGEJ  �          @�Q�@�\�	����ff�[(�C�u�@�\�@  ��z��~�HC�                                      BxzGS�  �          @�ff@AG��0  �\��� �C�޸@AG���{��33�H�\C��R                                    BxzGb�  �          @�{@O\)�hQ��{���C��@O\)�3�
�L(��Q�C��f                                    BxzGq<  �          @��\?У���
=����x��C���?У����H�.{��\)C��R                                    BxzG�  �          @���@`���aG����\�eC�w
@`���>�R������HC�Ф                                    BxzG��  �          @��\@�z��z�=p���
=C�\)@�z���Ϳ��\�]G�C��\                                    BxzG�.  �          @��\@�����=q���H����C�AH@����W
=�����G�C�\                                    BxzG��  �          @���@��ÿ�p�����7
=C��3@��ÿ�������|(�C��                                     BxzG�z  �          @��@�G���
�u�#33C�%@�G��Q�h���ffC��                                    BxzG�   �          @��@���=p��J=q��
C��{@����׿z�H�(Q�C�j=                                    BxzG��  �          @�=q@���=��
�˅���?s33@���?\)���R����@�G�                                    BxzG�l  �          @�=q@���=#�
��ff�8  >��@���>�33�}p��-G�@x��                                    BxzG�  T          @�{@�=q���Ϳn{��RC��{@�=q���Ϳ�G��,Q�C�p�                                    BxzH�  �          @�z�@�
=�������6�\C���@�
==�\)��=q�;�?E�                                    BxzH^  �          @��
@�
=����=�?��C�K�@�
=��녾�����HC���                                    BxzH!  T          @��@�  �#�
?���A?�C�4{@�  �1G�>�=q@:�HC�4{                                    BxzH/�  �          @�@�{�@��?@  AQ�C�#�@�{�Fff�L���{C�                                    BxzH>P  �          @�{@U�tz�˅����C���@U�I���+���ffC�T{                                    BxzHL�  �          @�{@e���Q쾸Q��s�
C�ٚ@e��mp�������G�C��
                                    BxzH[�  �          @�\)@xQ��q녾����=qC��q@xQ��^{��ff��z�C�3                                    BxzHjB  �          @�{@z�H�Y����  �XQ�C��=@z�H�6ff��R�ŮC��)                                    BxzHx�  �          @�
=@��\��Ϳ�ff�\��C�Ǯ@��\��Q����=qC�^�                                    BxzH��  �          @�\)@C33�vff�����p�C�0�@C33�=p��U��\)C���                                    BxzH�4  �          @���@:=q�|(��������C�:�@:=q�@  �`  ��RC��                                    BxzH��  �          @�  @9���vff�'����C��H@9���6ff�l(��(�HC��f                                    BxzH��  �          @�=q@2�\�xQ��7
=����C���@2�\�333�z�H�3
=C�o\                                    BxzH�&  �          @��\@"�\��
=�#33�ң�C��\@"�\�]p��s�
�&ffC�*=                                    BxzH��  �          @��@���p���qG�� �C��@�������Q��X��C��3                                    BxzH�r  
�          @��@��w��j�H�{C��@��!���{�T��C��\                                    BxzH�  �          @�ff@]p������p����
C��@]p��J�H�h���
=C��f                                    BxzH��  �          @���@n{���H��
=�ap�C�N@n{�j�H�.{��=qC���                                    BxzId  �          @��@J�H����У���z�C�c�@J�H�aG��8Q���Q�C��                                    BxzI
  �          @�G�@Q������!G���33C�~�@Q��Q��o\)�*�HC���                                    BxzI(�  �          @��R@
=���H�'
=�ڣ�C�\)@
=�a��z�H�.�C�t{                                    BxzI7V  �          @�ff?��������8����ffC�#�?����XQ���p��=
=C�aH                                    BxzIE�  �          @���@�����R�B�\��\C�@���A���
=�A�
C��
                                    BxzIT�  �          @��@,(��u��P  �=qC��@,(��&ff�����D�C���                                    BxzIcH  �          @���@E�p���?\)����C���@E�'
=�����3C��=                                    BxzIq�  �          @�33@7
=�q��:=q��Q�C���@7
=�)���}p��5�
C��R                                    BxzI��  �          @��@u�X���G���p�C�H�@u��R�N�R�33C��\                                    BxzI�:  �          @��@s�
�Q��%���
C��=@s�
�G��`  �{C���                                    BxzI��  T          @�G�@[��c33�=q��ffC�H@[��%��Z�H�Q�C�~�                                    BxzI��  �          @��R@<(����Ϳ�{���HC��f@<(��U��Fff��C��R                                    BxzI�,  �          @�Q�@]p��\�������
=C���@]p��{�[����C�8R                                    BxzI��  �          @���@z=q�N{�{��p�C�K�@z=q�z��H���
z�C��f                                    BxzI�x  �          @�(�@HQ��S33�P  �\)C���@HQ��z������?��C�C�                                    BxzI�  �          @���@XQ��'��u�'�C�)@XQ쿝p�����M
=C�H                                    BxzI��  �          @�=q@Vff�33��p��6�
C���@Vff�Q���{�V��C�%                                    BxzJj  �          @��H@J=q�j=q�'���G�C�e@J=q�'
=�j�H�'ffC�=q                                    BxzJ  �          @�G�@���������
C�` @�P���h���(�
C���                                    BxzJ!�  �          @���@,����=q�%��\)C���@,���@  �q��,��C��
                                    BxzJ0\  �          @���@1��Dz��c�
�"��C�q@1녿�p����
�T�C�
                                    BxzJ?  �          @��\@0�׿�Q���Q��YQ�C�L�@0�׽�G�����p�C��                                    BxzJM�  �          @��\@$z�Y�����
�s�C��3@$z�?L����(��t��A�G�                                    BxzJ\N  T          @��@,�Ϳ�(����f��C��
@,��>�����=q�r
=A=q                                    BxzJj�  �          @�(�@Y����p���
=�B��C�AH@Y��    ��
=�R�
=#�
                                    BxzJy�  �          @��@U����������V�RC�u�@U�?�p���z��L�A�ff                                    BxzJ�@  �          @�\)@>�R>�����=q�h��@��R@>�R?����ff�N\)BG�                                    BxzJ��  �          @��@-p�>����  �u
=A�
@-p�@33���\�U��BG�                                    BxzJ��  �          @��@]p�>���
=�I\)@�z�@]p�?����tz��0�
A��\                                    BxzJ�2  �          @�\)@U�!G����R�L33C�� @U?O\)���J(�AZff                                    BxzJ��  �          @�=q@q녾�
=�&ff�	z�C�ٚ@q�>��%��
@�                                    BxzJ�~  �          @�
=@@  ?����ff�M�A�  @@  @7��\(���RB.��                                    BxzJ�$  �          @��\@C�
?�=q��\)�G�RA��H@C�
@I���W���B7Q�                                    BxzJ��  �          @��@7
=@$z���Q��8B'�@7
=@qG��:=q��Q�BS�                                    BxzJ�p  �          @�\)@<(�?�������M
=A�33@<(�@8���XQ��ffB1�                                    BxzK  
�          @��@^�R@�  �L���	��BD(�@^�R@�G�?��@�
=BE�                                    BxzK�  �          @���@qG�@`  �����F�HB+�@qG�@k�=�?���B1(�                                    BxzK)b  �          @�G�@fff?�G��?\)�(�A�(�@fff@���Q���33A�\)                                    BxzK8  �          @��R@�
�#�
��G��C�� @�
?�ff��\)�t��B$��                                    BxzKF�  �          @�@�R?Q���(�\)A���@�R@!G���G��U�HBB(�                                    BxzKUT  �          @��@O\)?8Q���ff�[�AG
=@O\)@���p��9�
B��                                    BxzKc�  �          @�{@{�>�ff��p��@�\@��@{�?�Q��\)�)G�A�=q                                    BxzKr�  �          @���@>�R>\��ff�e�@陚@>�R?�p������Hp�B{                                    BxzK�F  T          @�Q�@;�?aG�����gz�A��@;�@   ��Q��?��B"=q                                    BxzK��  �          @���@G�?�
=���\�_�HAģ�@G�@Fff�����0�
B3{                                    BxzK��  �          @�33@5@��33�SG�B��@5@w��o\)��BV��                                    BxzK�8  �          @��@-p�@@  ��z��=�\B?��@-p�@��H�E���z�Bh
=                                    BxzK��  �          @Å@S�
@g��fff�
=B>G�@S�
@��������RBZ�                                    BxzKʄ  �          @���@ ��@���N{��Bq=q@ ��@�  ��p��d(�B��)                                    BxzK�*  �          @���@,��@����>�R��(�Bf�@,��@�����ff�L��Bwp�                                    BxzK��  �          @�p�@q�@o\)�O\)�	�B2�@q�@q�?��@���B4{                                    BxzK�v  �          @�ff@�G�>���l���G�@���@�G�?�
=�Tz��	�HA�z�                                    BxzL  �          @���@���<��
�{��-�>aG�@���?�33�k�� (�A���                                    BxzL�  �          @���@�\)>W
=�p���G�@@�\)?z�H���R��ffA)p�                                    BxzL"h  �          @��
?�\)�C�
�w
=�<z�C��{?�\)��  ��{�{�RC���                                    BxzL1  
�          @��׽���Q��{��33C�Lͽ������aG��$Q�C�!H                                    BxzL?�  �          @�33>�
=����s�
�#��C��f>�
=�.�R����vG�C�g�                                    BxzLNZ  �          @���?J=q�����QG���
C���?J=q�S�
��  �Y\)C��)                                    BxzL]   �          @�\)?^�R��=q�Z�H��RC�  ?^�R�R�\����]�C�l�                                    BxzLk�  T          @����Q���G��n{�ffC�o\��Q��:=q���H�p�
C�                                      BxzLzL  �          @�������{�X���ffC�˅���J�H���H�c  C���                                    BxzL��  �          @�>��
�hQ���\)�Q��C��=>��
�ٙ���
=\)C�b�                                    BxzL��  �          @�ff>����S�
��z��_=qC���>��ÿ�=q��  �C��3                                    BxzL�>  �          @��R=����n�R���
�L\)C�Ǯ=��Ϳ�������8RC��
                                    BxzL��  �          @�{�G��z=q��(��>��C�T{�G������  CyL�                                    BxzLÊ  �          @�p��Tz���p��q�� {C��{�Tz��1G����
�rQ�C}Y�                                    BxzL�0  �          @�(������z��G����Cy0������
=�����ffCw��                                    BxzL��  �          @�z������������HCx� �����=q�%����Cv�=                                    BxzL�|  �          @�������(�����}Cx�f����p��^�R���Ct�
                                    BxzL�"  �          @��
�O\)��=q��z��:p�C�g��O\)�����=qaHCy��                                    BxzM�  �          @�?(��  ��p�B�C���?(��#�
��33¨�=C�U�                                    BxzMn  �          @��;�  �r�\�����N{C�ᾀ  �����=q��C�H                                    BxzM*  
�          @\@U���33�\)�îC���@U��H���z=q�!��C�Z�                                    BxzM8�  �          @���@8������)���ϙ�C��@8���U�����-G�C��H                                    BxzMG`  �          @�G�?z�H���H�"�\��z�C�/\?z�H��G���=q�8z�C���                                    BxzMV  �          @�녾#�
�����|���'ffC��\�#�
�)����G��}z�C�9�                                    BxzMd�  �          @�=q�.{��=q�K����HC�/\�.{�b�\��=q�T�RC��3                                    BxzMsR  �          @�(������(��U���C|xR����S�
��z��U{Cu�                                    BxzM��  �          @���G��L����Q��N
=Cm�3�G���(����W
CS                                      BxzM��  �          @��0�׿����(��tz�CI���0��?z�H����v=qC ��                                    BxzM�D  �          @���#33��=q��(��w�COp��#33?B�\���8RC#T{                                    BxzM��  �          @���'���33��ff�g�
CT+��'�>��R��ff�|z�C-T{                                    BxzM��  �          @\��ff��33�^{�Q�C{^���ff�>{���_{Cr�                                     BxzM�6  �          @�p���33��Q��=q��C�ΐ33���R��G��2{C~�R                                    BxzM��  �          @�{������33�\)� Q�C�
�����1�����t{Cw�
                                    BxzM�  �          @��Ϳh���p�������L33C�1�h�ÿ�Q����H�=CqǮ                                    BxzM�(  �          @��8Q��Q�����c�C����8Q쿎{����(�Cl�f                                    BxzN�  �          @ȣ׾�����\����7�
C�P�����z���G��C�
=                                    BxzNt  �          @�p��8Q���=q�����>�C�H�8Q��z���
=  Cz�                                    BxzN#  �          @���{���\��(��.p�C��f��{������RB�C�
                                    BxzN1�  �          @�{�����j=q��  �IffCxn�������������Cc��                                    BxzN@f  �          @�\)�G��!G���
=�i{CgT{�G���Q�����  C>
                                    BxzNO  �          @�ff�)�������{�vG�CM�H�)��?^�R�����|C!�=                                    BxzN]�  �          @��'
=�}p���Q��|�\CH�q�'
=?�
=��
=�y�\C��                                    BxzNlX  �          @�{�-p��=p���  �{�CC33�-p�?�z����
�qQ�C�=                                    BxzNz�  �          @�p��^{�:�H����[��C?�{�^{?�  ��ff�UQ�C �                                    BxzN��  �          @��aG��5�����Y�C?p��aG�?�G���p��S�C G�                                    BxzN�J  �          @��^�R�����  �X{CD�
�^�R?s33��Q��Y�C$��                                    BxzN��  �          @�{�X�ÿE����H�^�C@� �X��?��R��  �X�\C�f                                    BxzN��  �          @���]p��.{�����\=qC?
�]p�?����p��T=qCE                                    BxzN�<  �          @�ff�U�333��p��b�C?Ǯ�U?�����G��Z  C�                                    BxzN��  �          @�ff�W��G������`p�CA��W�?�G������Z�Cp�                                    BxzN�  �          @����<(���������g�CL�=�<(�?@  �����p=qC%�                                     BxzN�.  �          @\�{���������s{C]@ �{>u��(���C-�q                                    BxzN��  �          @�(��   ��
=��(��w=qCQ�   ?B�\����=qC#�                                    BxzOz  �          @����������\)�{z�CU�H��?.{����=C#��                                    BxzO   �          @��H�(���=q�����33CP��(�?xQ���\)ǮCh�                                    BxzO*�  �          @˅�$zῠ  ���}=qCM��$z�?����
=33C�{                                    BxzO9l  �          @˅�$zῺ�H���
�y
=CQ�=�$z�?Y������C!�                                     BxzOH  �          @˅�(����
�����CRT{�(�?����33��C�{                                    BxzOV�  �          @��
��
=����G��\CAٚ��
=?����=qC�                                    BxzOe^  S          @�=q�녿0����{Q�CF���?�Q���\)z�C#�                                    BxzOt  �          @��H����\)����{CO����?��\���\#�C                                    BxzO��  �          @�(����\)�����3CRs3�?�G���33��C��                                    BxzO�P  �          @��
�E��G����Z�CU=q�E�>B�\����s�C0��                                    BxzO��  �          @�33�"�\��������|�CO���"�\?��\��
=�C0�                                    BxzO��  �          @�z��0�׿�p�����v��CL��0��?����z��y
=C��                                    BxzO�B  �          @�z��A녿�=q��  �n��CG�f�A�?�
=��\)�m33C�3                                    BxzO��  �          @�p��L�Ϳ\(�����k��CB�q�L��?����(��d=qC�                                    BxzOڎ  �          @�33�9��?#�
���u33C'���9��@0  ��\)�G33C��                                    BxzO�4  �          @���;�@2�\���H�Hz�CY��;�@�\)�U���
=B�=q                                    BxzO��  �          @��R�\@�����H�F�C\)�R�\@�p��]p��p�C B�                                    BxzP�  �          @θR�HQ�?��H��Q��[�\C�R�HQ�@x����G��C ��                                    BxzP&  �          @�G��HQ�?���ff�s�C)���HQ�@333��  �H
=C
(�                                    BxzP#�  t          @�33�C33>������H�x��C,s3�C33@-p���{�Op�C
c�                                    BxzP2r  �          @�33�N�R<��
��  �r�C3���N�R@z���Q��S�C@                                     BxzPA  �          @�z��B�\�fff����qffCDz��B�\?�z���ff�i�
C
                                    BxzPO�  �          @����L�Ϳ�{����g�CG.�L��?�33�����g{C (�                                    BxzP^d  �          @Ϯ�Fff�G���z��q�RCB+��Fff?�ff��
=�f��CaH                                    BxzPm
  �          @�z��5���  ����n�CO�R�5�?W
=���
�yQ�C#}q                                    BxzP{�  �          @���I�������h�HCHW
�I��?�\)��{�i�C c�                                    BxzP�V  �          @����c33��\)��ff�Y�CE�=�c33?�=q���R�Z�C#�                                    BxzP��  �          @˅�Fff��p������i33CI���Fff?�����k�C!{                                    BxzP��  �          @�33�N{��������e�
CG\)�N{?����33�eC ��                                    BxzP�H  �          @�z��C33�����  �n
=CG��C33?��R��
=�k�
C�H                                    BxzP��  �          @�
=�(Q�G����H��CDz��(Q�?������t�\C�f                                    BxzPӔ  �          @�p��.�R�^�R����}��CE�3�.�R?����33�r�HC�                                    BxzP�:  �          @�z��7���33���H�t
=CI���7�?�p���=q�r�Cٚ                                    BxzP��  �          @�33�5��p������x{CFaH�5�?�Q���  �p(�C\                                    BxzP��  �          @��H�%���R��  p�CA���%�?��
��  �q=qCQ�                                    BxzQ,  �          @���  ���H��(�G�C@0��  ?�(�����v��C
Ǯ                                    BxzQ�  �          @��H��p��+���\)aHCFǮ��p�?���
=k�C	)                                    BxzQ+x  �          @�33�녿:�H��
=\)CG��?������=C
u�                                    BxzQ:  �          @�G��=q��
=��p�CN
=�=q?��\�����C@                                     BxzQH�  �          @�(��(Q쿁G���
=��CI��(Q�?�Q���(��w��CJ=                                    BxzQWj  �          @��
�p��xQ���G��)CIz��p�?�G���p��|
=Cz�                                    BxzQf  �          @ʏ\�zῑ������
CN!H�z�?�{��
=ǮC��                                    BxzQt�  �          @�Q��!G��(���{
=CA���!G�?����q�CY�                                    BxzQ�\  T          @�G��*�H���
��L�C:���*�H@z������e�
C:�                                    BxzQ�  �          @�  �#�
�Tz���z���CF��#�
?˅��
=�tC8R                                    BxzQ��  �          @�  �&ff�������|\)CJ�f�&ff?�=q��Q��x(�C�q                                    BxzQ�N  �          @������u���R\)CJxR��?�G����H�~z�C!H                                    BxzQ��  �          @���(�þB�\��ff.C8\�(��@p���  �c�C�                                    BxzQ̚  �          @ȣ���
��(���z���C?��
@z������x�C�{                                    BxzQ�@  �          @�  ��\��{��z�G�C=����\@
=q��  �vffCk�                                    BxzQ��  �          @����þu��
=G�C<�)����@�\�����}G�B�Ǯ                                    BxzQ��  �          @��׿�G�>8Q������C-Y���G�@!������o  B�z�                                    BxzR2  �          @�(��$z�?�=q�����b(�C�{�$z�@j=q�c�
�  B�{                                    BxzR�  T          @�G�?^�R������Q��qC�4{?^�R?B�\��{{B$�H                                    BxzR$~  �          @�{��?p�����\�B�8R��@Mp�����\ffBƙ�                                    BxzR3$  T          @��ÿ�G�?��
���R.C�H��G�@HQ������G��B��                                    BxzRA�  �          @�=q�Z=q?   ����T�HC+���Z=q@=q�~{�.�C�=                                    BxzRPp  �          @�33�g�?=p������>z�C(k��g�@���Vff��HC��                                    BxzR_  �          @��H����������=qC78R��@  �����pffC(�                                    BxzRm�  �          @��H���>�{��  �qC)�׿��@(�����H�`B��H                                    BxzR|b  �          @����p�?�������Q�C���p�@dz����6��B�\                                    BxzR�  �          @�=q�ff?k���{(�C���ff@C33�����@�HB�=q                                    BxzR��  �          @�33�<��?G���{�iG�C%@ �<��@3�
��z��5�
Cc�                                    BxzR�T  �          @����:=q@ ������S�Ch��:=q@s�
�W
=���B��)                                    BxzR��  �          @�33��=q>����33C*����=q@#33�����g��B�\)                                    BxzRŠ  �          @�(���?�����ff�u\)CJ=��@W��xQ��-�B�B�                                    BxzR�F  �          @�
=��p�?
=q����fC����p�@>{�����e33B��                                    BxzR��  �          @���  >�=q��Q��C$녿�  @-p����\�qz�B�Q�                                    BxzR�  �          @��H���W
=���H(�CR� ��?�33���.C�q                                    BxzS 8  �          @��ÿ���p������tG�CrQ쿥��u��33  C6��                                    BxzS�  �          @�ff�B�\��ff��G��-�RC���B�\�33��ffW
Cy�{                                    BxzS�  �          @�{��ff�]p���ff�QC}
��ff�����\)aHCb�                                    BxzS,*  �          @��Ϳp���A���{�eQ�C|���p�׿
=�����CT!H                                    BxzS:�  "          @�33�W
=�p���ff�}��C{\�W
=<���  £G�C2�                                    BxzSIv  �          @��H�����"�\���\�r\)Cruÿ��ý�Q����3C7Ǯ                                    BxzSX  �          @��H��  ��33����RCl�
��  ?�����B�C�                                     BxzSf�  �          @������Ϳ�\)��=qQ�C}������?�Q����Bي=                                    BxzSuh  �          @��\�У׿���Q�8RCGQ�У�?�����R�C޸                                    BxzS�  �          @�=q�
=q�s33��{=qCp�
�
=q?�\)��G��qBخ                                    BxzS��  �          @��þB�\��=q���{C���B�\?�  ��=q�3B�.                                    BxzS�Z  �          @�33?n{��=q��� �C��?n{@������ffB�u�                                    BxzS�   
�          @����Ǯ�����33  C}33�Ǯ?�ff���k�B�#�                                    BxzS��  �          @��
�1G��H�����
�2{Cd���1G����\��=q�o(�CH(�                                    BxzS�L  �          @����#�
������
�W�HC_���#�
��\)��ff��C5�                                     BxzS��  �          @����`  �������C�CO^��`  >�33����X33C.B�                                    BxzS�  �          @�����=q��
�z�H�"�\CNٚ��=q�����=p�C5ٚ                                    BxzS�>  �          @�Q��s33�fff�����E�CA^��s33?�Q���\)�A��C"��                                    BxzT�  �          @�ff�333��p���=q�Z33CW=q�333>�{��ff�v  C-                                      BxzT�  �          @����\�33��\)�i\)C]ٚ��\>�Q����
aHC++�                                    BxzT%0  �          @���z���������\�HCa�q�z�����
L�C4�
                                    BxzT3�  �          @���
=�$z���ff�U��Ccn�
=�k���3333C9�\                                    BxzTB|  �          @����p���0  �[��(�CX.�p�׿s33��G��?\)CB8R                                    BxzTQ"  �          @������H��
�l�����CN�q���H�W
=��\)�7�HC6�H                                    BxzT_�  �          @��\�~{����y���){CJh��~{>����
=�:�HC033                                    BxzTnn  �          @�=q��\��Q���(���C}!H��\?B�\��33 �=B�\                                    BxzT}  �          @���?^�R��=q�����C���?^�R?5���
=B
=                                    BxzT��  �          @��ͽ�G��33��\)��C�k���G�?����=q§�B�(�                                    BxzT�`  �          @�{���\��ff����L�Cf�=���\?�=q���33C�                                     BxzT�  �          @�  �Q녿�������Ck�=�Q�?\��\)�{B왚                                    BxzT��  �          @�ff�녿z�H��G���Co����?��H���
p�B�33                                    BxzT�R  �          @�(��#�
�����
=.Co�H�#�
?�=q��33�B�Ǯ                                    BxzT��  �          @��
�.{�\(���\)Q�Cg޸�.{?����Q���B�k�                                    BxzT�  �          @�(��&ff�\)��G�£
=C\�{�&ff@���BָR                                    BxzT�D  �          @�(��(��W
=��L�Ci�(�?�ff��ff��Bُ\                                    BxzU �  �          @�\)�k��{����\C���k�>\���R«�B�{                                    BxzU�  �          @�\)>u�/\)��\)�tz�C��>u�W
=��­��C��R                                    BxzU6  �          @�\)?�\)�J=q�����R��C�� ?�\)�J=q���R��C�{                                    BxzU,�  �          @���@%��P  �~{�.�
C�/\@%���\)��Q��r�HC�E                                    BxzU;�  �          @��H?�  �A���
=�\�
C�4{?�  �z����\B�C��R                                    BxzUJ(  �          @�
=�   �
�H����  C���   ?�����¥��C5�                                    BxzUX�  �          @��ÿ(�ÿ������L�Cy�q�(��?p����(��fB�u�                                    BxzUgt  �          @��þ�
=����33ffC�H���
=?+���{¤B�.                                    BxzUv  �          @�녿#�
�ff��(�k�C|�q�#�
?.{��
=¢8RCu�                                    BxzU��  �          @����
=q�������.C����
=q>�p���  §� CT{                                    BxzU�f  �          @\=#�
�
=q��p�8RC��=#�
?#�
��G�§��B��\                                    BxzU�  �          @��H>��Ϳ�Q���\)��C���>���?^�R���¡�HB�{                                    BxzU��  �          @��H?.{�-p����
�z(�C��?.{<�����§(�@p�                                    BxzU�X  �          @\>��R�A���  �o�C��=>��R��������«��C��3                                    BxzU��  �          @\?��\�AG���33�g�C�Y�?��\��p����33C�#�                                    BxzUܤ  �          @��
?O\)������(�C�f?O\)?��\���RW
BO{                                    BxzU�J  
�          @Å?5���H���
#�C��?5?����z�aHB{(�                                    BxzU��  �          @��H>����p���(�ǮC��q>���?(�����§.B~��                                    BxzV�  T          @\?�G����
��\)#�C�N?�G�?�p���.BF(�                                    BxzV<  �          @\>k���=q��=q�\C��>k�?��H��p�8RB���                                    BxzV%�  �          @\�k������\)�Cp�{�k�?���33�C #�                                    BxzV4�  �          @�33�L�Ϳ��
���HCr� �L��?�����z��RB�z�                                    BxzVC.  �          @Å�aG��Y����\)�fC_Ǯ�aG�?��R��ff#�B�                                    BxzVQ�  �          @�z�\)�&ff��=q£��Ce:�\)@p���{B�B�B�                                    BxzV`z  �          @��H�����G��\C�T{��?0����(�¦k�B��                                    BxzVo   �          @\?!G��S33�����az�C�j=?!G������  £p�C�                                      BxzV}�  �          @���?��\�,(����H�v�C�\)?��\=��
��\) ��@��                                    BxzV�l  �          @�z�?\)������C���?\)>���\¦=qB{                                    BxzV�  �          @�(�?�=q�Dz����
�cz�C��R?�=q��Q����R  C���                                    BxzV��  �          @�(�?}p��)����33�xffC�5�?}p�=���
=¡{@ٙ�                                    BxzV�^  �          @�{���*�H����~\)C�녿�>8Q���33¨C"�3                                    BxzV�  �          @�{�&ff��\����Q�C|k��&ff?^�R��=q��B��{                                    BxzVժ  �          @�p��#�
�N�R���H�_  C�]q�#�
��R��G�¢k�C_�R                                    BxzV�P  �          @\�!G��a���G��T��C���!G��fff��(�Ck
                                    BxzV��  �          @�33����� ������\)C�� ����>�Q��\­
=B��                                    BxzW�  �          @���=L���%�����#�C��\=L��>�z��Å®ffB��\                                    BxzWB  �          @��
����5��z��x�HC������    ���H±B�C5�q                                    BxzW�  �          @��Ϳz��L����p��gffC��q�z��
=���¦� CX@                                     BxzW-�  �          @��Ϳ�  �,�������nG�Cp���  =u��p�z�C1��                                    BxzW<4  �          @ȣ��Mp���z���p��S(�CR�=�Mp�?����  �h\)C)^�                                    BxzWJ�  �          @�G��J=q�^�R���H�i�RCCW
�J=q?�(����
�[��Ck�                                    BxzWY�  
�          @�{?Tz��Mp���{�[33C�AH?Tz�#�
�����C�)                                    BxzWh&  �          @ƸR?���;���Q��j=qC�G�?���������
C�C�                                    BxzWv�  �          @�  >�=q����(�  C���>�=q?^�R��{£33B���                                    BxzW�r  �          @�G��L�Ϳ�Q���=qB�C�\�L��?�����G��HB�u�                                    BxzW�  �          @��>�\)��=q��(��fC��>�\)?�Q���G�G�B��                                    BxzW��  �          @��H>#�
��{��aHC��>#�
?���\)aHB��                                     BxzW�d  �          @ə�?(�ÿ��
���
k�C�Y�?(��?��H��z�8RB���                                    BxzW�
  �          @��>u�}p���¡8RC���>u@G���p�B��q                                    BxzWΰ  �          @�녾��ý��Ǯ­�\CHT{����@4z�����{�RB�L�                                    BxzW�V  �          @��>\��
=���#�C��>\?��������B��                                    BxzW��  �          @�{?J=q>�  �ȣ�¤�A�z�?J=q@J�H����kffB��                                    BxzW��  �          @Ϯ>��R��=q��ff¬ffC���>��R@+����Hp�B��R                                    BxzX	H  T          @�
=���
��
=�ƸR8RC������
?�����Q�\)B�=q                                    BxzX�  �          @�\)��  �U���Q��kG�C�ٚ��  ������¬��Ch�q                                    BxzX&�  �          @Ϯ�W
=�L�����
�q=qC�'��W
=�#�
�θR¯Q�CZ��                                    BxzX5:  �          @�{��Q��1���G���C�ὸQ�>�\)��p�®�B�                                      BxzXC�  �          @�?�ff��(����R.C��H?�ff?�ff��=qG�B4{                                    BxzXR�  �          @θR?����W���=q�^��C�.?��׾�ff�����C�
                                    BxzXa,  �          @�
=?�ff�+������o(�C��?�ff>�=q���\A{                                    BxzXo�  �          @�\)?˅�7
=�����m�C���?˅=�G��Ǯff@���                                    BxzX~x  �          @У�?��H�=q��ff�u�\C���?��H?z���p�Q�A��                                    BxzX�  �          @�G�@7��fff���R�y  C�Q�@7�?���ff�f��BQ�                                    BxzX��  �          @�
=?��
��G���33�HC�H�?��
@ ����33��BA�R                                    BxzX�j  �          @�  ?�z῝p���
=W
C��
?�z�?�  ���H�B*Q�                                    BxzX�  T          @�Q�@$z�z�H��
=��C��3@$z�?�{��  �p
=B�\                                    BxzXǶ  �          @�Q�?�p���ff��
=ffC�Ff?�p�@���
=Q�BW��                                    BxzX�\  �          @�Q�?��ÿ�����=qC�
?���?�����
=�\BL��                                    BxzX�  �          @�Q�@�R�������Hp�C��)@�R@'
=��Q��hz�BF(�                                    BxzX�  �          @љ�?�Q쾅���G�� C���?�Q�@1G�����p�\Bj�                                    BxzYN  �          @���?�z�>��H��33��A��H?�z�@]p����H�\G�B��R                                    BxzY�  �          @�Q�?�33>\)��G�33@�=q?�33@G����R�j�\B��3                                    BxzY�  �          @У�?�\)>�����
=AAp�?�\)@U������XffBrff                                    BxzY.@  �          @�  ?У�>u�����=A=q?У�@N{�����a
=B|=q                                    BxzY<�  �          @�  ?��
>����
=��@��R?��
@G
=��z��a�Bp�R                                    BxzYK�  �          @Ϯ?�{�����Ǯ��C���?�{@G���  �qB`                                      BxzYZ2  �          @�Q�?�33����ȣ��C���?�33@��  �\Ba                                      BxzYh�  �          @У�?��W
=�ə��C��)?�@���33Bh��                                    BxzYw~  �          @�G�?�녾����(�\C���?��@5���
=�tp�B�                                    BxzY�$  �          @���?ٙ���G��ə��qC�7
?ٙ�@;����\�j�HBo�                                    BxzY��  �          @���?���>k��ȣ�L�@�=q?���@Mp���z��^Q�Bqz�                                    BxzY�p  �          @�G�?��>�=q��  �AG�?��@O\)��33�\  Bo(�                                    BxzY�  �          @��?��
>�z��ə�AQ�?��
@Q���(��\�Bv{                                    BxzY��  �          @��?���?@  ��=q8RA�z�?���@l�����PB�                                    BxzY�b  �          @�  @
�H<#�
�\��>�  @
�H@;���=q�^33BU�                                    BxzY�  �          @У�@��=���������@\)@��@@����  �W{BM�                                    BxzY�  �          @�  @ff?
=q��(��Ae@ff@Z�H����OffBj�                                    BxzY�T  �          @�\)?�33?h����33��A�(�?�33@o\)����F
=B|Q�                                    BxzZ	�  �          @Ϯ?G�@G��Å�B���?G�@�
=��(��)��B�#�                                    BxzZ�  �          @�ff?(��?��
��
=��B�#�?(��@��H���R�;�HB���                                    BxzZ'F  �          @�ff?�G�@
=��  aHB���?�G�@�  ��\)�$\)B�                                    BxzZ5�  T          @�(�?}p�>�
=��Q� z�A��
?}p�@Y�������`�\B��                                    BxzZD�  �          @˅?aG�>��R����£\A��R?aG�@S�
���H�ep�B��                                    BxzZS8  �          @˅>���?(������¥��Bl��>���@g����[p�B��                                    BxzZa�  �          @˅=�Q�>L���ʏ\¯�B�#�=�Q�@O\)��{�lB���                                    BxzZp�  �          @�(�?
=>���ʏ\©  Ahz�?
=@L�����R�l�HB��                                    BxzZ*  �          @˅>�z�?^�R����£L�B��>�z�@s�
��=q�T  B�33                                    BxzZ��  �          @��>L��?u��\)¢
=B�Ǯ>L��@w
=��\)�P��B��                                    BxzZ�v  �          @ƸR>�=q�\��«Q�C�ٚ>�=q@*=q���HǮB�G�                                    BxzZ�  �          @�
=?+�?Q��\ B�BKQ�?+�@j�H��p��SQ�B�\)                                    BxzZ��  �          @�ff?���Z=q���C�C��{?���Tz������qC���                                    BxzZ�h  �          @�{>�(�����������C���>�(�?ٙ�����B��3                                    BxzZ�  �          @�Q�>�ff�   ��ff§�C��>�ff@p���{�B�(�                                    BxzZ�  �          @�z�>�p��������
¬�C�}q>�p�@(�����R�vB��)                                    BxzZ�Z  T          @��R�W
=?�p������RB��W
=@�
=��Q��+�Bʊ=                                    Bxz[   �          @�\)�s33@7
=��z��n�B�녿s33@�  �Fff����Bɞ�                                    Bxz[�  �          @��>W
=>�ff���\©�fB���>W
=@G
=�����`��B�p�                                    Bxz[ L  �          @�(��(��@]p���(��R��Bɞ��(��@����
=��G�B�G�                                    Bxz[.�  �          @�녿�33@w
=���8\)B�Q쿓33@����\��(�B��                                    Bxz[=�  �          @�=q��(�@���[��
=B�q��(�@�(��n{��HB�.                                    Bxz[L>  �          @�(�����@�
=�4z���B�B�����@�  �#�
�У�Bس3                                    Bxz[Z�  T          @�33�p��@L����G��\ffBԽq�p��@�(��'
=���HB�Ǯ                                    Bxz[i�  �          @�녿^�R@=p���p��g�Bԣ׿^�R@���5���Bǽq                                    Bxz[x0  �          @��׿��
@#�
�����uQ�B��쿃�
@�ff�I�����B̸R                                    Bxz[��  �          @�z�@Q�?5��33�{ffA��@Q�@?\)�xQ��5�RBM�
                                    Bxz[�|  �          @�33>���?���������B���>���@e����\�B=qB�B�                                    Bxz[�"  �          @��ÿ��
@����r�\�%p�B݊=���
@������W�B�                                    Bxz[��  
�          @���?k�?�G����H�)B>\)?k�@XQ��|���A�RB�\)                                    Bxz[�n  �          @��R?��H?n{��\)k�B�\?��H@W���33�Cp�B��                                    Bxz[�  �          @�
=�#�
?��R�����{B�z�#�
@�=q�g���HB�#�                                    Bxz[޺  �          @�G��L��@ ����z��{z�B�k��L��@�ff�P  �	{B�Q�                                    Bxz[�`  �          @�녾\@�R�����BŨ��\@�\)�Vff���B�L�                                    Bxz[�  �          @��
��@Z=q�����Y�HB�� ��@�=q�\)����B��H                                    Bxz\
�  �          @�33<�@N�R��(��b�B��<�@��R�*�H��G�B�W
                                    Bxz\R  �          @��\>B�\@Z�H��
=�X�B���>B�\@�G�����p�B��H                                    Bxz\'�  �          @��\?#�
@^�R���H�Qz�B�L�?#�
@�G���\��=qB�G�                                    Bxz\6�  �          @���?h��@h����z��E��B��?h��@��\������B��R                                    Bxz\ED  �          @�33>W
=@s�
���D�B�>W
=@����p����RB���                                    Bxz\S�  �          @��<��
@k���{�I�B�L�<��
@�z���
����B��                                    Bxz\b�  �          @��׾��H@=p���p��j��B�녾��H@���5���p�B�G�                                    Bxz\q6  T          @�p����@\)��G��|��B̮���@����J=q�=qB�aH                                    Bxz\�  �          @�����@O\)���
�Z��B�
=����@�33����\)B���                                    Bxz\��  �          @�Q���@��H�\)�0��B�
=���@��׿�(��m�B���                                    Bxz\�(  �          @�{�u@33��p�ǮB��q�u@�G��W����B���                                    Bxz\��  �          @��׿�G�@QG����H�UQ�B�=q��G�@�33�����G�B�L�                                    Bxz\�t  �          @�(���=q@�p��l(��  B�\)��=q@�ff��z��6{B�#�                                    Bxz\�  �          @��Ϳ�@���p�����B���@�
=��(��?�Bؙ�                                    Bxz\��  �          @�z��{@��R�U�
�B�B���{@��׿5��ffB�.                                    Bxz\�f  �          @�{��z�@�G��C�
���B�(���z�@������<(�BԊ=                                    Bxz\�  �          @�z��G�@���7
=��{B�L��G�@��þ����(�B�=q                                    Bxz]�  �          @���z�@��\�8Q����B����z�@��
����B�z�                                    Bxz]X  �          @���=q@�=q�8����=qBݮ��=q@��
�\)���B�.                                    Bxz] �  �          @��Ϳ\@�33�>�R��G�B��Ϳ\@��L�Ϳ�(�B��f                                    Bxz]/�  
�          @�Q쿂�\@�
=�AG�� �B�zῂ�\@�33�����@��BȽq                                    Bxz]>J  �          @�\)@��	�������V�HC�P�@�>u����@�p�                                    Bxz]L�  �          @�ff@\)��H���\�Y(�C�\)@\)>#�
��8R@�G�                                    Bxz][�  �          @��>�?0�����R¦B��>�@Z�H��(��V
=B��3                                    Bxz]j<  �          @����R?�Q���G�ffB�녿�R@\)����6  B���                                    Bxz]x�  �          @��\���
?c�
���¢(�B�Ǯ���
@dz���\)�M�HB��3                                    Bxz]��  �          @�G�?�?   ��(��\A�(�?�@N{�����V��B�                                      Bxz]�.  �          @���?У׾���
=� C��f?У�@�\��  �r��BZ{                                    Bxz]��  �          @��?˅��33��\)�HC��)?˅@����{�n��Ba�                                    Bxz]�z  �          @��?���?8Q���=qu�B�\?���@XQ�����NQ�B�{                                    Bxz]�   �          @��
>L��?˅�����HB�\)>L��@�(���G��1�\B�\)                                    Bxz]��  �          @�{�
=@G���{W
B��f�
=@���fff��HB�L�                                    Bxz]�l  �          @���?�?z�H��=qz�Bv�H?�@g
=����HffB�k�                                    Bxz]�  �          @�
=?�ff?
=q��G�u�A�p�?�ff@Mp���G��R�B���                                    Bxz]��  �          @�p�>#�
>W
=���
®�BU33>#�
@=p������h��B���                                    Bxz^^  �          @���?8Q�?�p����\B�Bn�?8Q�@u�����=p�B���                                    Bxz^  �          @��\?��H?�Q����\��Bhz�?��H@��\�j�H�{B���                                    Bxz^(�  �          @���?^�R?����ǮB�aH?^�R@���u��&  B��                                    Bxz^7P  �          @���?�?�����(���B~Q�?�@n{����D��B��3                                    Bxz^E�  �          @��?�
=?��
��33�Bap�?�
=@�{�qG��#ffB��                                    Bxz^T�  �          @�=q��?�����¤.C���@W���{�W�HBŊ=                                    Bxz^cB  �          @�G�>\)?�R���R§Q�B�Ǯ>\)@W�����X\)B�L�                                    Bxz^q�  �          @���?!G�����ff§aHC��=?!G�@333��
=�o�B���                                    Bxz^��  �          @�z�@G��:�H�y���:33C���@G�������\�C���                                    Bxz^�4  �          @�p�?fff��G���p���C��?fff?˅���\33BqG�                                    Bxz^��  �          @�
=?��
�����C��3?��
@������B��)                                    Bxz^��  �          @�Q�?��ͿB�\��  ��C��3?���@z�����Bw��                                    Bxz^�&  �          @�G�?&ff���
������C���?&ff?����\��B��
                                    Bxz^��  �          @��?:�H�Ǯ��
=��C���?:�H?�{����ǮBv33                                    Bxz^�r  �          @�Q�?\(��n{���\�C�Z�?\(�?��H���
=B��H                                    Bxz^�  �          @���?�=q����{aHC��?�=q@.{��  �n��B��                                    Bxz^��  �          @�G�?�z�?@  ����A�\)?�z�@Y����ff�I�B�{                                    Bxz_d  �          @���?333?(���¡\)B#z�?333@Vff��(��V(�B�8R                                    Bxz_
  �          @�����ff?�p���=q��BոR��ff@�����33�4��B���                                    Bxz_!�  �          @��ÿG�?�����HB�B��׿G�@qG���G��@�\B�\)                                    Bxz_0V  �          @�=q��  ?У���ffǮB�
=��  @���z�H�)�\Bսq                                    Bxz_>�  �          @�녿�ff?�Q���=q�Ck���ff@s33����<ffB���                                    Bxz_M�  �          @���=��
�E����R£��C�=��
@�\���
�{B�\)                                    Bxz_\H  �          @�Q�>�Q쿠  ���H��C���>�Q�?�Q���\)  B�G�                                    Bxz_j�  �          @�  ��=q>�=q��p�¬G�C5þ�=q@C33��G��e�B�                                      Bxz_y�  �          @�
=�^�R?�������fB�녿^�R@n{��
=�?ffB�Q�                                    Bxz_�:  �          @�  ��=q?E���£��B�#׾�=q@_\)�����Q�RB�                                      Bxz_��  �          @����ff?������C޸��ff@l�����
�;�RBӨ�                                    Bxz_��  �          @���?�z�
=��33L�C��
?�z�?�33��=q�w��BE�R                                    Bxz_�,  �          @��@?\)��p�����H�RC�9�@?\)>����
=�e��@�(�                                    Bxz_��  �          @��\@G
=�{�y���1��C��=@G
=�W
=����_�RC�)                                    Bxz_�x  �          @�33@=p��
�H��p��B��C�� @=p�>B�\��ff�f��@e                                    Bxz_�  �          @��@*=q�����  �h{C�޸@*=q?�(������j
=A��H                                    Bxz_��  �          @��H@#33�������\�o��C��@#33?�z���Q��i��A��
                                    Bxz_�j  �          @��@HQ쿳33��p��P�C���@HQ�?z�H�����X\)A��R                                    Bxz`  T          @��@e���\)�w
=�.��C�<)@e�>�z������G�R@���                                    Bxz`�  �          @��
@0�׿�����Q��W
=C�G�@0��?(����=q�n�AVff                                    Bxz`)\              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxz`8              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxz`F�   �          @���?˅�(����H�sQ�C�  ?˅?   ��G�A���                                   Bxz`UN  �          @�G�?��#33�����]p�C��?�<���ff�{?�                                     Bxz`c�  �          @�33@U��mp��
=�ʸRC��@U��33�x���3�C�*=                                   Bxz`r�  �          @�p�@i���X���,����C���@i����{�����5G�C��                                   Bxz`�@  �          @�\)@j�H�HQ��G��p�C��f@j�H����Q��?p�C�'�                                   Bxz`��  �          @��@W��:=q�mp��Q�C��)@W��.{��p��Vp�C�Ff                                   Bxz`��  �          @���@U��5��s33�$  C��\@U�����ff�YG�C�)                                   Bxz`�2  �          @��H@E�0  ��=q�2  C�+�@E��p���z��f(�C��{                                   Bxz`��  T          @�(�@C33�9����33�0��C�>�@C33����  �i
=C��                                    Bxz`�~  �          @�(�@<(��{��{�D��C���@<(�=u���\�o��?�\)                                   Bxz`�$  �          @�z�@   �\)��=q�\�\C��@   >�G���G���A!�                                   Bxz`��  �          @��@{�����Q��h��C�b�@{?Y�������~��A�{                                   Bxz`�p  �          @���@p��������\�
C��{@p�>�
=�����=A\)                                   Bxza  �          @�Q�@U����p��<�C�N@U>���Q��a(�@(�                                   Bxza�  �          @���@c�
���{�<z�C���@c�
>Ǯ��z��WQ�@ƸR                                    Bxza"b  �          @���@\(�����ff�?(�C�&f@\(�>�Q���p��[�@�G�                                    Bxza1  �          @�{@.{�n�R�aG��G�C��@.{��  �����fC���                                    Bxza?�  �          @�{@I����G��>{���C���@I����\���\�J�RC���                                    BxzaNT  �          @�  @s�
�qG��(����p�C��@s�
��(�����0p�C�S3                                    Bxza\�  �          @�Q�@xQ��`���9�����C���@xQ��{��  �5ffC��q                                    Bxzak�  �          @��@z�H�7��W
=�
�\C���@z�H�Tz����H�={C��                                    BxzazF  T          @��@����j�H�G�C��q@��    ��
=�3�\C���                                    Bxza��  �          @��R@�  ����J=q� �C���@�  ��33�u��!\)C��=                                    Bxza��  �          @��R@�
=�(��Z�H�Q�C���@�
=��(����2p�C�q                                    Bxza�8  �          @�ff@���%�P  ��HC�E@���#�
����/33C��\                                    Bxza��  �          @�  @|(��J�H�Mp���\C���@|(�������:��C�                                    BxzaÄ  �          @�Q�@�p��\(��&ff��\)C�9�@�p��ٙ��}p��%Q�C��                                     Bxza�*  �          @��@w
=�\)����ffC��@w
=��|���$��C�b�                                    Bxza��  �          @�  @|����33������
C��
@|���(���j=q�ffC�"�                                    Bxza�v  �          @���@�  ����W
=� ��C��=@�  �L���2�\��\)C�~�                                    Bxza�  �          @\@����
=�����RC��)@���i���
=���C�
=                                    Bxzb�  �          @\@�z���ff����z�C�  @�z��e�(���z�C�Y�                                    Bxzbh  �          @�=q@c�
��z���H����C�H@c�
��Q��5�߮C�Ǯ                                    Bxzb*  �          @���@u��G����
�p�C�)@u�[��H������C�{                                    Bxzb8�  T          @���@�\)��G��u�
=qC�H�@�\)�l���
�H����C�e                                    BxzbGZ  
�          @�Q�@��
���
����!G�C��=@��
�j�H�����33C�*=                                    BxzbV   �          @�\)@�  ����p��aG�C��@�  �j�H� ����(�C��q                                    Bxzbd�  �          @�p�@�33��Q�\)��33C��3@�33�hQ���R��
=C�9�                                    BxzbsL  �          @�@|����������"=qC�E@|���N�R�A����
C�^�                                    Bxzb��  �          @�33@�����\)����33C���@����g��(����C��                                    Bxzb��  �          @��@x�����\��
=�c�
C��{@x���5�P���33C��                                    Bxzb�>  T          @��@���}p��8Q���C��@���E�%���p�C���                                    Bxzb��  �          @��@���qG�������{C�� @���\)�P  �=qC�g�                                    Bxzb��  �          @��@~{�|(���{�3
=C�� @~{�8Q��9�����HC�f                                    Bxzb�0  �          @�=q@u���H=#�
>�G�C���@u�s33�ff���
C���                                    Bxzb��  	�          @�\)@aG��l(��	�����RC���@aG��
=q�l(��'(�C�E                                    Bxzb�|  T          @�  @��\�c33������C�P�@��\�AG���{���C��                                    Bxzb�"  T          @��
@�33�HQ�@,(�Aܣ�C�Y�@�33����?O\)@��RC���                                    Bxzc�  "          @�@�G��4z�@S33B(�C��@�G����\?�p�Ag�C�Y�                                    Bxzcn            @���@w��^{@"�\AЏ\C��@w�����?�\@���C�R                                    Bxzc#            @�=q@u��p��@�RA��RC��@u����
=���?p��C��q                                    Bxzc1�  g          @�=q@a����
@A�(�C�K�@a����
�W
=�z�C���                                    Bxzc@`  
�          @��@L(����@�\A���C�<)@L(���Q쾮{�Y��C��                                    BxzcO  s          @�  @fff���H?�{A2{C��{@fff���\���;�C��H                                    Bxzc]�  �          @�  @\������?Q�A�
C���@\�����H��  �r�\C�@                                     BxzclR  
�          @�p�@C33��z�?���A8  C���@C33���H����S
=C���                                    Bxzcz�  
�          @��H@7���{?Tz�A��C�� @7������=q��=qC�N                                    Bxzc��  �          @��H@=p����R?��A���C��@=p���z�W
=�\)C�Ff                                    Bxzc�D  
�          @�  @Vff�y��@��A��
C�\)@Vff��33>aG�@��C�                                    Bxzc��  
�          @���@Z�H�C33@`  Bz�C�!H@Z�H��(�?ǮAz=qC�H                                    Bxzc��  
�          @���@l(��Q�@@��A�33C�0�@l(����\?��\A"ffC�4{                                    Bxzc�6  "          @��@h���L��@C33B �\C�P�@h������?�=qA.�RC�*=                                    Bxzc��  
�          @�  @n�R�5@S�
B�RC�^�@n�R���H?��RAq�C�,�                                    Bxzc�  �          @�  @j=q�=q@l��B �
C�K�@j=q�z�H@33A�\)C���                                    Bxzc�(  
�          @��
@g����H@���B7(�C�  @g��O\)@2�\A�\C��                                    Bxzc��  �          @�\)@Mp���  @��BP=qC�Z�@Mp��>�R@QG�B
=C��                                    Bxzdt  "          @��@\(��H��@!�A�(�C���@\(��~{?+�@陚C�t{                                    Bxzd  T          @�(�@@����    �L��C�R@@���w
=�
�H��33C��
                                    