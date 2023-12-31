CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230422000000_e20230422235959_p20230423021751_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-04-23T02:17:51.038Z   date_calibration_data_updated         2023-04-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-04-22T00:00:00.000Z   time_coverage_end         2023-04-22T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxzd*�  
�          @�=q@Q��|���	����Q�C��f@Q�����s�
�A�
C�ff                                    Bxzd9f  �          @�ff@   �>�R�k��6�
C���@   �L����p�p�C��                                    BxzdH  T          @�\)@ff��G��z���33C��@ff�����  �G��C�AH                                    BxzdV�  
�          @�=q@����  ��p���Q�C���@���0  �s33�8��C���                                    BxzdeX  �          @��@�\�����ff��{C�Ф@�\�L(��u�5�
C�K�                                    Bxzds�  �          @�G�@   �����\)�G�C�z�@   �fff��ˮC�]q                                    Bxzd��  
(          @��R@Z�H�u�>�ff@�
=C��@Z�H�c�
��p���33C���                                    Bxzd�J  �          @��@*=q�\)����zffC��H@*=q�5��G
=��C���                                    Bxzd��  
�          @��
@(Q��|�Ϳ���\)C��{@(Q��#�
�a��/��C��=                                    Bxzd��  g          @�=q@����{���
�k�
C�'�@���B�\�J=q���C�o\                                    Bxzd�<  A          @���?���������33C�?��>�R�mp��9�C�aH                                    Bxzd��  "          @���@����ÿTz��ffC�W
@��c�
�;��p�C�+�                                    Bxzdڈ  �          @���>aG�����1G����C�� >aG���\��ff�z��C���                                    Bxzd�.  �          @�z�u��  �L(��{C�Ff�u��Q���\)��C�w
                                    Bxzd��            @�ff?Y�����
�*=q����C���?Y���!���ff�l{C�K�                                    Bxzez  A          @���?aG���=q�
�H��  C�l�?aG��<(�����U�C�J=                                    Bxze   �          @�Q�=������.�R�\)C��
=��33�����y\)C���                                    Bxze#�  
�          @�G�@W��n{��Q�}p�C�)@W��Mp������C�7
                                    Bxze2l  
�          @�z�@Q��xQ�?�\@�ffC��@Q��h�ÿ�Q����RC��                                    BxzeA  �          @�p�@Tz��aG�?:�HA
=qC���@Tz��[�����HQ�C��                                    BxzeO�  �          @�\)@?\)�]p�?�33Ab�RC�k�@?\)�e��(�����C���                                    Bxze^^  
�          @�Q�@R�\�Vff=#�
?�C�8R@R�\�=p���=q���C�f                                    Bxzem            @���@P  �j=q>�{@{�C���@P  �W
=��p���{C���                                    Bxze{�            @�ff@N{�p  =L��?(�C�N@N{�S�
��\����C�
                                    Bxze�P  
�          @��@`���R�\�����C�n@`���(Q���\���
C��{                                    Bxze��  
�          @�33@e��J�H�G��z�C�=q@e�����{�ٙ�C�R                                    Bxze��            @�(�@Y���@�׿fff�3
=C�AH@Y������\)��\)C���                                    Bxze�B  A          @���@^�R�Mp��L���=qC��f@^�R��H�  ���
C���                                    Bxze��  g          @��@N{�W
=����M��C���@N{���#33��C�|)                                    Bxzeӎ  s          @�(�@E�[������  C���@E����@�����C��q                                    Bxze�4  �          @�p�@W
=�U�������C�� @W
=�G��2�\�
33C���                                    Bxze��  �          @���@QG��Vff������C�'�@QG��z��K���
C���                                    Bxze��  �          @��@W��Vff�\���C��@W��p��<(����C�\)                                    Bxzf&  A          @���@e�R�\��z��W33C���@e�z��&ff��z�C���                                    Bxzf�  	�          @�Q�@Z�H�Z=q���
�l��C���@Z�H�Q��0���
=C���                                    Bxzf+r  �          @��\@w
=�Fff����PQ�C��q@w
=�
�H�\)��
=C�T{                                    Bxzf:  T          @�G�@l���QG��p���,��C�Ff@l���=q�����C�o\                                    BxzfH�  
Y          @��@j�H�Vff�fff�$Q�C��\@j�H� ��������C���                                    BxzfWd  h          @�=q@}p��C33�E����C�+�@}p���
����ȏ\C��q                                    Bxzff
  �          @�\)@W��c�
�333�=qC��)@W��2�\�z���p�C�5�                                    Bxzft�  �          @�\)@Z=q�`�׿E��33C�
@Z=q�-p��
=��G�C��                                     Bxzf�V  T          @�
=@aG��XQ�aG��#\)C�
@aG��#33�����Q�C�\                                    Bxzf��  "          @�z�@a��S33�5�C�y�@a��#�
�����=qC��                                    Bxzf��  
�          @��@`���QG��:�H�
=C���@`���!G������Q�C�&f                                    Bxzf�H  T          @��R@W��aG��p���.�HC�޸@W��)��� ������C��\                                    Bxzf��  �          @���@g��Vff�s33�-�C���@g��\)����C��R                                    Bxzf̔  
�          @���@n�R�K������IC��@n�R�G���R����C�Q�                                    Bxzf�:  "          @�G�@~{�?\)�^�R�\)C��@~{�p��(����C�q�                                    Bxzf��  
�          @��@Y���c�
�E��p�C���@Y���1G�����
=C�s3                                    Bxzf��  @          @��R@fff�U�+���G�C���@fff�'�����
=C��3                                    Bxzg,            @�G�@z�H�C�
�G���RC��)@z�H�z�����ə�C���                                    Bxzg�  
�          @��@~�R�E�����C��@~�R�p�������C��                                    Bxzg$x  r          @���@j=q�R�\��G��8��C���@j=q��H�����C�@                                     Bxzg3  h          @��@j�H�Vff�p���+�C���@j�H�   ��H��ffC�޸                                    BxzgA�  r          @���@3�
���
������C�.@3�
�W���H��\C��                                    BxzgPj  "          @���@1�����\����C��R@1��_\)��
�ڸRC�B�                                    Bxzg_  
�          @�Q�@;���  ��G�����C�  @;��Tz�����C���                                    Bxzgm�  �          @�Q�@,(���{�k��(Q�C�XR@,(��e�(���  C�q�                                    Bxzg|\  
�          @��\?�����=q?�(�A�G�C���?�����\)�^�R�p�C�T{                                    Bxzg�  �          @��?����G�?Y��AG�C�� ?�����Ϳ�33��\)C�                                    Bxzg��  "          @�p�@����?
=@�p�C���@����׿�p����C�Ff                                    Bxzg�N  �          @�p�@�
����>�G�@�C�
@�
��  �����ffC��R                                    Bxzg��  �          @��H@���z�
=q�У�C���@��X������p�C�~�                                    BxzgŚ  
�          @�(�?�33���
����33C�c�?�33�XQ������{C���                                    Bxzg�@  
�          @��?�
=�������RC�h�?�
=�hQ��z���{C�                                    Bxzg��  "          @�p�@ ����z��R���RC��@ ���W
=�   �33C�l�                                    Bxzg�  T          @��@
=q�|�Ϳ����V=qC�U�@
=q�>�R�333�{C��{                                    Bxzh 2  "          @���?��
����
�H��{C�XR?��
�$z��tz��OC�l�                                    Bxzh�  
�          @�{?W
=��Q��,(��Q�C���?W
=����Q��q\)C�%                                    Bxzh~  �          @�\)?�p�����%���z�C�g�?�p������e�C���                                    Bxzh,$  "          @��R?��\)�,(��ffC�0�?��G���  �k�HC��                                    Bxzh:�  �          @���@�w�������
C�1�@��H�mp��D��C�q�                                    BxzhIp  
�          @��@���p���33�ȣ�C��f@���
=�dz��>(�C��                                    BxzhX  
�          @��H@��dz�������C�Ff@��  �Y���:�
C���                                    Bxzhf�  
�          @�z�@-p��c33�������C��\@-p��Dz��ff��33C��)                                    Bxzhub  �          @�\)@��hQ�=#�
>��HC��@��N�R��z����HC�|)                                    Bxzh�  
�          @�(�@>�R�Z=q�O\)�#
=C���@>�R�(Q���
����C�G�                                    Bxzh��  "          @�G�@#33�c�
��\)��p�C�Ǯ@#33���Tz��0�C�R                                    Bxzh�T  �          @�p�?����Tz��P  �%�RC��H?��Ϳ���p�Q�C�33                                    Bxzh��  
�          @��H=�G��������wC�H�=�G������H±� C��R                                    Bxzh��  "          @��R?
=q�333�^�R�J�C�q�?
=q�\(���z���C���                                    Bxzh�F  "          @�  ?z�����y���q��C���?zὣ�
��ff¤��C�\                                    Bxzh��  T          @��R?녿��
��z��HC�9�?�?\)���
8RB2ff                                    Bxzh�  "          @��R@G��U?�  AEC��f@G��Z�H�(���{C�,�                                    Bxzh�8  
�          @�\)@H���.{@%�A���C���@H���g
=?��
AAC���                                    Bxzi�  
�          @��@9���)��@B�\B�
C��\@9���p��?�(�A�p�C���                                    Bxzi�  �          @�\)@0  �\)@Tz�B'�C���@0  �o\)?�ffA��C�!H                                    Bxzi%*  �          @��@G
=�=p�@
=A�  C�7
@G
=�mp�?8Q�A�C���                                    Bxzi3�  �          @��@XQ��S�
?@  A(�C��@XQ��Q녿\(��&{C���                                    BxziBv  �          @���@S�
�Vff?�ffAv�\C�U�@S�
�dz�Ǯ���
C�k�                                    BxziQ  "          @���@W
=�XQ�?W
=A ��C�o\@W
=�XQ�O\)�C�ff                                    Bxzi_�  T          @��@XQ��U>�@��C���@XQ��K������ZffC�ff                                    Bxzinh  "          @��@J�H�c33?0��Az�C���@J�H�^{���\�C\)C�:�                                    Bxzi}  
�          @��\@P  �c33>�{@��\C�<)@P  �S�
��=q��C�<)                                    Bxzi��  �          @��R@^�R�I��=�?�
=C��@^�R�7
=�������C�S3                                    Bxzi�Z  T          @�\)@Mp��[������  C��=@Mp��5��   ���C�H�                                    Bxzi�   �          @���@U��XQ�\��33C�G�@U��5�������C���                                    Bxzi��  @          @���@XQ��U���Q쿋�C��
@XQ��:�H������(�C��R                                    Bxzi�L  
�          @��\@b�\�K�?
=q@�\)C��@b�\�E��z�H�<(�C�~�                                    Bxzi��  "          @��@c�
�8Q�?��AP(�C�|)@c�
�C33�\���C��{                                    Bxzi�  T          @��H@QG��2�\?��RA�\)C��@QG��J=q=u?333C���                                    Bxzi�>  h          @�=q@G
=�'�@�\A�
=C��{@G
=�P��?�R@��\C��3                                    Bxzj �  	�          @�Q�@e��
�H@2�\B�C�g�@e��L��?��RA�(�C��                                    Bxzj�  @          @�(�@|����
@4z�A�
=C��@|���U?���A{\)C��)                                    Bxzj0  
�          @��@p  �J=q?�
=A��C��=@p  �l(�>�\)@@  C���                                    Bxzj,�  �          @��\@e��<(�@(��A��
C�J=@e��tz�?��\A0(�C��                                    Bxzj;|  
Z          @��@XQ��/\)@:�HBz�C�}q@XQ��p��?�{Ap(�C��3                                    BxzjJ"  
�          @��@W
=�!G�@>�RB\)C��
@W
=�e?�G�A���C���                                    BxzjX�  
�          @��@Z�H�#�
@I��B�RC���@Z�H�mp�?�33A���C�W
                                    Bxzjgn  	�          @���@W
=�   @P  BG�C���@W
=�l��?�\A��
C�                                      Bxzjv  T          @�@[��0��@L��B
=C��{@[��z=q?���A�33C��                                     Bxzj��  
�          @��\@W��5@Z�HB=qC���@W����\?�\A��C��                                    Bxzj�`  
�          @�z�@S�
�+�@l(�B#�\C���@S�
����@�A���C���                                    Bxzj�  
�          @��R@W
=� ��@:�HBG�C�� @W
=�c�
?�(�A��C���                                    Bxzj��  
�          @�\)@[��7�?\A�=qC��@[��P  =u?B�\C�AH                                    Bxzj�R  "          @�ff@\(��AG�?�ffAO�C�\)@\(��J�H��
=��33C���                                    Bxzj��  
�          @�ff@33�(�@�  Bl�HC���@33�@S33B4p�C�AH                                    Bxzjܞ  
Z          @�\)@'����\@p��BT�C�]q@'��&ff@9��BC���                                    Bxzj�D  T          @��R@<(����@H��B'��C��=@<(��E?��HA���C���                                    Bxzj��  	�          @�ff@(���p�@L��B+�\C��@(���Z=q?�\)A��C���                                    Bxzk�  
(          @��@:=q�{@.�RB��C�O\@:=q�Mp�?���A���C�R                                    Bxzk6  �          @��@8Q��%�@'�B\)C�3@8Q��^�R?�Q�Ak�C�Ф                                    Bxzk%�  "          @���@R�\����@C33B$�
C��@R�\��@{A噚C��                                    Bxzk4�  
�          @�@Z=q����@Q�A���C�3@Z=q�3�
?�G�A}�C�<)                                    BxzkC(  
Z          @�ff@^�R�%�?�G�A�G�C��{@^�R�Fff>�(�@�  C�'�                                    BxzkQ�  T          @�\)@W��2�\?޸RA��C�7
@W��QG�>��
@z=qC��                                    Bxzk`t  "          @��@w
=�*�H?�\)A��\C���@w
=�@  =#�
?   C�                                    Bxzko  �          @�{@��
�{?���AS\)C��@��
�.{��Q쿂�\C�N                                    Bxzk}�  �          @�=q@���,��?�  Ad��C�U�@���>�R���
�\(�C��                                    Bxzk�f  �          @��H@����<(�?xQ�A0��C��f@����Dz��(���z�C�Q�                                    Bxzk�  T          @�=q@tz��7
=?@  A�RC��q@tz��8�ÿ
=���C�p�                                    Bxzk��  	`          @�  @���/\)>�ff@��C�S3@���)���O\)���C���                                    Bxzk�X  �          @���@�{�z�>.{@   C�4{@�{�
=q�^�R��C��                                    Bxzk��  
�          @���@�
=�{>aG�@'�C��\@�
=�z�c�
�'�
C��)                                    Bxzkդ  �          @�p�@w
=�'�>�\)@Z�HC���@w
=��R�h���2�RC��=                                    Bxzk�J  
�          @�33@~�R�
=q��\)�c�
C�� @~�R���ÿ��H�vffC��q                                    Bxzk��  	�          @��@u�#33����(�C�/\@u�z�˅���C���                                    Bxzl�  
�          @�33@g
=�.{�333���C�y�@g
=��ÿ�����{C���                                    Bxzl<  T          @�z�@ ���vff��{����C��@ ���S�
�   ��C���                                    Bxzl�  T          @�{@z�H�녿�  �G33C��@z�H��{��33���C�ٚ                                    Bxzl-�  
�          @�Q�@q��,�Ϳ�{�W\)C�33@q녿�Q����ٙ�C�aH                                    Bxzl<.  �          @�=q@Fff�K������Y�C�&f@Fff����
=���\C�4{                                    BxzlJ�  "          @��@L(��H�ÿ���Q�C���@L(�����
��G�C���                                    BxzlYz  
�          @�=q@Z�H�333���
�QG�C�XR@Z�H�z��	����C�ff                                    Bxzlh   
�          @��H@P���L(��Ǯ��p�C��3@P���,�Ϳ޸R��
=C�33                                    Bxzlv�  
�          @��
@���vff?!G�@��RC��q@���o\)�����_33C�Z�                                    Bxzl�l  
F          @��
@���z�H?Y��A)G�C��3@���y���p���=p�C�f                                    Bxzl�  
n          @�z�@1��j�H    ��C��\@1��S�
������Q�C�                                    Bxzl��  
          @�ff@0���q�>.{@�C�R@0���^�R��p����C�:�                                    Bxzl�^  T          @�ff@j=q�1녿^�R�,(�C�]q@j=q�Q���R��C��                                    Bxzl�  �          @�
=@vff�$z�c�
�-C�%@vff��
=��z����C��                                    BxzlΪ  �          @���@s�
�1녿:�H�z�C���@s�
��Ϳ�{����C��                                    Bxzl�P  �          @�  @s�
�.{�J=q�C�=q@s�
�
=�������C��H                                    Bxzl��  h          @�\)@dz��A녿#�
��G�C���@dz��p������  C���                                    Bxzl��            @��R@y������  �Ep�C�T{@y�����H����z�C�%                                    Bxzm	B  �          @�
=@�녿�����������C�,�@�녿�Q�����G�C��3                                    Bxzm�  �          @���@�{��\)�.{��C���@�{������33C��                                     Bxzm&�  "          @�
=@����(��.{�C�^�@����Q쿙���l��C��                                    Bxzm54  T          @�Q�@w��+�?\)@�33C���@w��*=q�#�
���C��R                                    BxzmC�  
�          @�Q�@|(��{>W
=@"�\C��)@|(��z�^�R�,(�C�Ǯ                                    BxzmR�  T          @�Q�@]p��(��	���י�C�o\@]p����
�?\)��C��
                                    Bxzma&  �          @���@fff�1G��Y���)�C�9�@fff��ÿ������
C���                                    Bxzmo�  
�          @�  @hQ��1G���\)�\Q�C�T{@hQ����(���z�C�g�                                    Bxzm~r  T          @�{@l���,(��^�R�+�
C��)@l���z��
=��  C�j=                                    Bxzm�  �          @�@p  �7
=��{����C�U�@p  � �������C���                                    Bxzm��  T          @�33@tz��2�\��  �>�RC��@tz��ff�����C��R                                    Bxzm�d  �          @��@xQ쿋��2�\�
�C�.@xQ�>L���>�R�  @:=q                                    Bxzm�
  �          @���@��H��Q��0����C�S3@��H���G
=��HC�3                                    Bxzmǰ  
�          @�  @��R���333� �C��\@��R�����H���C�P�                                    Bxzm�V  �          @��H@xQ쿧��@���{C���@xQ�=��
�Q�� ?�Q�                                    Bxzm��  �          @��R@`�׿xQ��Vff�*�C�K�@`��?
=q�\���0�\A33                                    Bxzm�  T          @�@XQ�&ff�^{�4C��=@XQ�?^�R�Z�H�1��Af�\                                    BxznH  "          @�ff@a녿G��U��*��C�� @a�?0���Vff�+�HA1��                                    Bxzn�  �          @�
=@q녿�
=�>�R�{C�XR@q�>B�\�L���!  @3�
                                    Bxzn�  �          @�
=@z�H�xQ��7����C��@z�H>����@����R@�                                      Bxzn.:  
�          @�
=@�z���H��
���
C��f@�z�0���%��p�C�G�                                    Bxzn<�  
�          @�
=@������
��\)C�
@���&ff�$z����HC���                                    BxznK�            @���@�p����	����=qC�>�@�p��@  �-p��ffC��                                    BxznZ,  �          @���@�33�����R��
=C��@�33�8Q��3�
�	�C��
                                    Bxznh�  
�          @��@j�H�L���J=q� G�C��)@j�H?
=�Mp��#
=Aff                                    Bxznwx  
n          @���@G
=>\�z�H�MG�@�\)@G
=?�p��Y���*��B�R                                    Bxzn�  
�          @��@=p��#�
��(��Y\)C��{@=p�?�
=�q��?��A�(�                                    Bxzn��  	�          @��\@Fff�#�
�|���M�C�#�@Fff?�ff�w
=�F�RA�Q�                                    Bxzn�j  
Z          @��R@W
=����|���E\)C��@W
=?��H�r�\�:�
A�ff                                    Bxzn�  
�          @�33@W��
=�u��@��C��@W�?���n�R�:33A��                                    Bxzn��  �          @�=q@\(��Q��g��6��C�U�@\(�?@  �hQ��7z�AE��                                    Bxzn�\  
�          @�Q�@L�Ϳ#�
�s33�E=qC�b�@L��?}p��n{�?�A�G�                                    Bxzn�  �          @��R@HQ�(�����\�OffC��@HQ�?�=q�\)�I
=A�                                      Bxzn�  
Z          @�ff@7���  ��=q�affC�u�@7�?�ff�����L��A�
=                                    Bxzn�N  
Z          @��@j�H���H�9����C���@j�H=�Q��H���"=q?���                                    Bxzo	�  T          @�z�@����"�\>\)?��C��R@����Q�k��2ffC��
                                    Bxzo�  	�          @�  @e��/\)?���A���C�N@e��S33?(��@�\)C��                                    Bxzo'@  	�          @�
=@|���.�R?�@أ�C���@|���.{������HC��R                                    Bxzo5�  
�          @���@s33�4z�?�=qA��C���@s33�N{>�z�@Q�C���                                    BxzoD�  
(          @��@`  �N�R?���A���C���@`  �l(�>�33@z=qC��                                     BxzoS2  6          @��R@S33�c33?fffA'�
C�n@S33�g
=�!G���\C�4{                                    Bxzoa�  
x          @�=q@S�
�h��?k�A)�C�#�@S�
�l�Ϳ#�
��(�C��=                                    Bxzop~  "          @��@dz��S33?��
Ak�C�� @dz��a녾.{��p�C��                                    Bxzo$  �          @��@^{�fff?�{AG�C���@^{�o\)��ff���C�g�                                    Bxzo��            @�33@u�%?��HA�\)C�H@u�J=q?=p�A�C�E                                    Bxzo�p  "          @�@x���*�H@Q�A�Q�C��f@x���S33?^�RAffC��
                                    Bxzo�  6          @�@w��"�\@�
A�{C�aH@w��P  ?��AA�C���                                    Bxzo��  
�          @��@j=q�+�@�\A��C��q@j=q�XQ�?�G�A733C���                                    Bxzo�b  �          @���@�z��  ?޸RA�=qC��H@�z��0��?0��@�(�C�*=                                    Bxzo�  "          @��\@�z��{>aG�@%�C��@�z����
=q��\)C�k�                                    Bxzo�  
�          @���@�Q��33��  �8z�C���@�Q쿫���
=���HC��H                                    Bxzo�T  T          @�G�@�{�����  �f{C��@�{�8Q��
=��(�C��)                                    Bxzp�  
(          @�(�@��׿��Ϳ����C�H�@��׿@  ���ӅC�Q�                                    Bxzp�  T          @�z�@�ff�G���  �6ffC�aH@�ff��  ����;�C���                                    Bxzp F  T          @�p�@�  ��p��L����RC���@�  �޸R�z�H�.�\C��R                                    Bxzp.�  
�          @�{@�ff�@  ��
=�T��C��@�ff�aG�����y�C��q                                    Bxzp=�  T          @�Q�@��׾#�
��=q��p�C�q@���>���G���33@���                                    BxzpL8  �          @�{@��
=�G�����
=?��@��
?E���p���  A
=                                    BxzpZ�  �          @�ff@��?333��p��[
=@��R@��?��׿Y����HAK�                                    Bxzpi�  �          @���@���?�Q��+����
A�G�@���@�ÿ�����HA�                                    Bxzpx*  "          @�  @��?�G�����Ǚ�A��H@��@�׿�
=�}p�A�                                    Bxzp��  
�          @��@��?�{?&ff@�=qAC\)@��?B�\?��
A4��A\)                                    Bxzp�v  T          @�(�@QG�?�(��S�
�.{A�
=@QG�@�H�$z�� �HB�                                    Bxzp�  "          @�z�@   ?����{�[��A�G�@   @Dz��O\)�33BK�                                    Bxzp��  
�          @��H@@��?�Q��tz��C33A��H@@��@5��<(��B,                                    Bxzp�h  	�          @�G�@.{?�Q��|���PQ�A�
=@.{@8Q��Dz��(�B:Q�                                    Bxzp�  �          @��R@0  ?�(��{��R�A�p�@0  @*=q�HQ��=qB0=q                                    Bxzp޴  "          @�p�@tz�?!G��9���=qA�@tz�?��H��H��Q�A���                                    Bxzp�Z  6          @�{@���?E��/\)�
=A,Q�@���?��
�{��ffA�Q�                                    Bxzp�   �          @���@~�R>���B�\�=q@��@~�R?�=q�/\)�p�A�G�                                    Bxzq
�  r          @���@��>�Q��-p��
=@�p�@��?���
=�ޣ�A�p�                                    BxzqL  "          @�\)@�
=>����#�
��p�@�33@�
=?��\�\)�י�A�ff                                    Bxzq'�  �          @���@vff>���N{�ff@{@vff?���:=q��A�ff                                    Bxzq6�  "          @��H@�G�>��H�G��ȸR@�G�@�G�?��R����ffA��R                                    BxzqE>  
�          @��
@*�H�  ��33��  C��R@*�H����'��$�
C��3                                    BxzqS�  
�          @��@b�\�(Q�xQ��E�C��R@b�\�33��z����C���                                    Bxzqb�  T          @���@Mp��W�?}p�A@��C���@Mp��_\)�\��(�C�J=                                    Bxzqq0  
�          @�
=@%�e@p�A�z�C���@%���?(��@�p�C��{                                    Bxzq�  T          @�ff@,(����?ٙ�A��
C��@,(���
=�����C�Ф                                    Bxzq�|  �          @���@5��p�@�
A��HC�(�@5��\)?�@���C��                                     Bxzq�"  �          @�Q�@"�\��=q@�RA��C�5�@"�\��33>��@�\)C�Ф                                    Bxzq��  �          @���@33��=q@�AΏ\C��@33����?
=q@��C��R                                    Bxzq�n  S          @�=q?��R��Q�@$z�A�  C��?��R����?&ff@�ffC���                                    Bxzq�  �          @�G�@
=��33@{A�z�C�=q@
=��
=?!G�@���C���                                    Bxzq׺  T          @�33@%���  @!�AׅC���@%���z�?8Q�@��C���                                    Bxzq�`  
�          @���@N�R��  @z�A�(�C�t{@N�R���\?�R@˅C��
                                    Bxzq�  �          @��
@N{�l(�@.�RA��C���@N{��ff?�\)A9G�C���                                    Bxzr�  �          @��
@R�\�qG�@"�\A�G�C��=@R�\���R?k�A\)C�9�                                    BxzrR  �          @��@l���R�\@%A�33C�,�@l������?�z�A?33C�H�                                    Bxzr �  
�          @�33@O\)�w�@=qA��C���@O\)���?E�@���C���                                    Bxzr/�  T          @�33@<���}p�@%A�33C�S3@<������?fffAG�C�4{                                    Bxzr>D  
Z          @��
@`���l(�@Q�A�ffC�Ǯ@`�����?Q�AC��\                                    BxzrL�  �          @�z�@c33�hQ�@(�A�33C�5�@c33����?fffA�RC��)                                    Bxzr[�  �          @��H@Mp��xQ�@=qA�z�C��\@Mp���  ?G�A Q�C�                                    Bxzrj6  �          @��H@Mp��u@{A�=qC���@Mp���\)?Y��A�C��                                    Bxzrx�  �          @��@g
=�j=q@��A�G�C�U�@g
=���R?+�@�p�C�P�                                    Bxzr��  
�          @��@b�\�}p�?�  A{\)C��f@b�\�����Q�p��C���                                    Bxzr�(  �          @��\@AG���  @p�A�p�C��=@AG���(�?L��A�C���                                    Bxzr��  �          @�33@P  �vff@��A�33C�R@P  ��\)?W
=A
=qC���                                    Bxzr�t  �          @��\@_\)�w
=?�p�A���C�{@_\)��=q>��@�ffC�|)                                    Bxzr�  �          @�33@e��u�?���A�\)C��f@e�����>Ǯ@���C���                                    Bxzr��  
�          @��
@X���{�@33A�=qC�g�@X�����>�ff@�(�C��=                                    Bxzr�f  �          @��\@\���~�R?�=qA��C�u�@\�����
>k�@�C�"�                                    Bxzr�  �          @��H@e�}p�?�=qA�C�q@e��Q�<#�
=�C��                                    Bxzr��  "          @��
@u��U�@��AɅC�y�@u��~�R?}p�A#�C��3                                    BxzsX  �          @��@vff�W�@�A�ffC�ff@vff�~�R?c�
AC��                                    Bxzs�  �          @��@u��Z�H@A�(�C��@u�����?n{A�HC��R                                    Bxzs(�  �          @��
@o\)�hQ�@33A��C��\@o\)��(�?z�@��RC�
                                    Bxzs7J  
o          @��@n{�g
=@
�HA��C��@n{����?5@�RC��                                    BxzsE�  
�          @���@b�\�^�R@p�A���C��f@b�\��(�?��\A*�\C�L�                                    BxzsT�  
�          @�  @333��G�@A˙�C�Y�@333���?5@�\)C��q                                    Bxzsc<  T          @�\)@>{�x��@�HA�=qC���@>{��  ?W
=Ap�C��{                                    Bxzsq�  
�          @��R@C�
�q�@(�A�33C�z�@C�
���?fffA��C�`                                     Bxzs��            @�
=@9���qG�@*=qA�C���@9����\)?�\)A<��C�z�                                    Bxzs�.  T          @���@K��O\)@G
=B	C�>�@K���z�?�p�A���C�                                    Bxzs��  T          @���@U����@j=qB&��C�q@U��c�
@#33A�
=C���                                    Bxzs�z  
�          @��@aG���  @�  B933C�z�@aG��5@L(�B  C���                                    Bxzs�   �          @�  @[��e@{A���C��R@[�����?G�A  C�˅                                    Bxzs��  
Z          @�=q@�Q��&ff@.{A�33C��H@�Q��Y��?�{A��
C��3                                    Bxzs�l  �          @�=q@~�R�2�\@'
=A�=qC�y�@~�R�b�\?�Q�ArffC�*=                                    Bxzs�  �          @��H@hQ��O\)@.{A陚C��@hQ���  ?���AeC�q                                    Bxzs��  "          @��@7���33@!G�Aՙ�C�xR@7���\)?c�
A{C���                                    Bxzt^  
�          @�(�@   ��
=@(�A�
=C���@   ��ff>�ff@�p�C�`                                     Bxzt  	�          @�G�@���p�@=qA�Q�C���@���Q�?E�A��C�
=                                    Bxzt!�  
(          @���@,(��a�@6ffB  C���@,(����?�Ayp�C���                                    Bxzt0P  
�          @�  @��r�\@?\)B\)C��@���33?��HA�33C��                                    Bxzt>�  T          @��?�{�Z�H@P  B(�C�G�?�{��33?�=qA���C��R                                    BxztM�  
(          @��@����@uBG33C�n@��i��@.�RBp�C��                                    Bxzt\B  T          @�  @Vff�G�@|(�BC�C�g�@Vff��@Z=qB"�HC���                                    Bxztj�  
�          @���@HQ�
=q@��BR�C�&f@HQ���H@l��B4\)C���                                    Bxzty�  �          @�@�  ?O\)@��A��A#33@�  ��Q�@"�\A�33C�u�                                    Bxzt�4  T          @�
=@�Q�?�Q�@��A�{A�{@�Q�?z�H@:�HBffAC�                                    Bxzt��  T          @��\@s33@O\)@(�A�(�B!��@s33@(�@Z�HBA�p�                                    Bxzt��  �          @�=q@:=q@�p�@\)A�=qB\=q@:=q@G�@c33B  B<{                                    Bxzt�&  
Z          @��
@'
=?��@   B	��A�33@'
=?�@�B$�RA8Q�                                    Bxzt��  "          @�p�?���(�?�ffA�p�C��H?���
==��
?k�C��{                                    Bxzt�r  T          @�(�?�G�����?���Ah��C��?�G����R����p�C��\                                    Bxzt�  
�          @�33?�\)����?�\)AA�C��)?�\)���
�333��C��H                                    Bxzt�  T          @�{?����z�?W
=A33C��=?�����
�}p��)�C��3                                    Bxzt�d  �          @�p�?�p����
?�G�AW�
C��?�p���Q�   ��Q�C���                                    Bxzu
  "          @���?�{����?��RAP��C�!H?�{���Ϳ\)��(�C���                                    Bxzu�  �          @��?�p�����?^�RA
=C���?�p���G��Tz���C��=                                    Bxzu)V  
Z          @�
=?����33?5@���C�S3?�������}p��1��C�j=                                    Bxzu7�  �          @�z�@����z�?���AK�
C�G�@����Q��G���p�C��3                                    BxzuF�  �          @���@*�H��p�?�
=AN�\C��\@*�H��녾�����G�C�+�                                    BxzuUH  �          @��@L����  ?���A��C�z�@L����G�=u?z�C��{                                    Bxzuc�  T          @�\)@Mp���=q?޸RA�  C�%@Mp���p�>�  @&ffC��                                    Bxzur�  T          @�\)@O\)��
=?��RAR=qC��@O\)��zᾅ��0��C�9�                                    Bxzu�:  
�          @�{@L(���(�?\A�C��q@L(�����<�>��
C���                                    Bxzu��  
(          @��@c33�{�?��A2ffC��@c33������33�k�C��)                                    Bxzu��  T          @���@Z=q�|��?��A�\)C�aH@Z=q���=�G�?���C�c�                                    Bxzu�,  �          @�z�@g��qG�?�(�AR�\C���@g��}p������\C�33                                    Bxzu��  
�          @�G�@tz��[�?�p�AW
=C�@tz��i��    <��
C�(�                                    Bxzu�x  �          @�z�@}p��=q?�33A�(�C�U�@}p��/\)?�\@���C���                                    Bxzu�  
�          @�p�@|�����?���A���C�)@|���5?.{A (�C�#�                                    Bxzu��  �          @��@��H�\)?�G�A��C���@��H�,��?h��A)�C�AH                                    Bxzu�j  �          @��H@&ff��\)?�AK\)C�{@&ff����Ǯ��\)C��R                                    Bxzv  T          @�ff@P����=q?�Q�Av�HC�S3@P����=q    <��
C��H                                    Bxzv�  T          @���@N�R��
=?���A��C��R@N�R��  =��
?Y��C��{                                    Bxzv"\  "          @���@u��c�
?�p�A�ffC��@u��|(�>�ff@�{C��                                    Bxzv1  
�          @��@a��xQ�?���A�
=C�%@a�����>�(�@�\)C��                                    Bxzv?�  �          @��@�z��%�?�{A���C�y�@�z��=p�?+�@��C��R                                    BxzvNN  
�          @��@�z��@��?��AbffC��=@�z��Q�>�\)@8Q�C���                                    Bxzv\�  "          @�33@��\�W�?}p�A#�C��@��\�`�׾.{���
C�y�                                    Bxzvk�  
�          @��
@��R�A�?�33Ahz�C��@��R�Tz�>���@W
=C��=                                    Bxzvz@  
�          @�33@�(��:�H?�
=AB�HC���@�(��H��>#�
?��C���                                    Bxzv��  
(          @��@�=q�
=q?\A�=qC���@�=q�"�\?:�H@�C�q                                    Bxzv��  �          @�G�@�33��H?�\)A�(�C�#�@�33�8��?z�HA#33C���                                    Bxzv�2  �          @��
@��?\)?��HA��
C�  @��XQ�?#�
@��HC�S3                                    Bxzv��  T          @�z�@��R�@  ?�
=A�z�C��@��R�XQ�?(�@�
=C�h�                                    Bxzv�~  "          @��
@�ff�Vff?У�A�33C���@�ff�l��>�@�C�Q�                                    Bxzv�$  �          @�(�@�  �<��?�z�A�\)C�\)@�  �U�?�R@ə�C��)                                    Bxzv��  
�          @��H@�\)�P��?�\)A���C�,�@�\)�g
=>�@�z�C�                                    Bxzv�p  "          @���@��H�@��?�{A��C���@��H�W
=?\)@�G�C�q                                    Bxzv�  T          @���@�=q��\?�ffA_\)C���@�=q�ff?z�@�z�C�                                    Bxzw�  �          @��\@�ff�
=q?�(�AS\)C���@�ff�(�>�@�C�L�                                    Bxzwb  �          @�33@��
�=q?��
A2=qC�8R@��
�'
=>L��@
=qC�AH                                    Bxzw*  
�          @�33@����I��?�  A���C��q@����^{>�
=@��C���                                    Bxzw8�  �          @��\@�33�p��#�
���C���@�33�z�L���Q�C��)                                    BxzwGT  �          @�(�@�
=�2�\>��@3�
C�  @�
=�.�R�#�
��  C�J=                                    BxzwU�  �          @�  @���QG�?��RAP��C�1�@���_\)>\)?��HC�C�                                    Bxzwd�  �          @�@e�o\)?��\A2�HC���@e�w��W
=��
C�n                                    BxzwsF  T          @���@E��U��R���C�W
@E��>{�����ffC��                                    Bxzw��  T          @�@(���R���
=qC���@(��
=�:�H�L��C��)                                    Bxzw��  �          @��@Vff��p�@Dz�B�\C�` @Vff�&ff@�A�G�C�3                                    Bxzw�8  i          @��R@l�Ϳ�\)@Dz�B�C���@l����@#�
A�=qC��)                                    Bxzw��  
          @��@����\?���A��C��@�����?���A\Q�C�G�                                    Bxzw��  "          @�(�@�=q�G�?�(�A��C�\@�=q�"�\?��RAT��C�z�                                    Bxzw�*  �          @�p�@�{�(�?�=qA�C�y�@�{�$z�?Q�A
�RC��q                                    Bxzw��  �          @�@�����?�ffA2ffC��
@���&ff>�  @(Q�C��)                                    Bxzw�v  �          @�p�@��R�)��>\@�  C�Q�@��R�(Q����p�C�b�                                    Bxzw�  �          @���@�z��.�R�\)����C��@�z��#33�z�H�'
=C��R                                    Bxzx�  T          @��\@�  �4z�>�=q@1G�C���@�  �0�׿�R���C��                                    Bxzxh  
o          @���@��<(�?.{@أ�C��
@��@�׾���%�C�N                                    Bxzx#  �          @�  @����G�<#�
=uC���@����>{�u� Q�C�aH                                    Bxzx1�  q          @�ff@���N�R��ff��\)C��q@���;����r{C��                                    Bxzx@Z  
=          @�33@h���0�׾Ǯ���
C�k�@h��� �׿�(��vffC���                                    BxzxO   �          @�{@�  �P  �z�H�&�HC�J=@�  �2�\��
=���\C�W
                                    Bxzx]�            @�p�@����Mp���\)��33C��R@����#�
�!G���z�C���                                    BxzxlL  T          @��H@����AG���Q��zffC��q@�������\�˅C���                                    Bxzxz�  
�          @��@s�
�=p���(���C�3@s�
���33��\)C��                                    Bxzx��  "          @�  @q��6ff�\��z�C�~�@q��  ��
���C���                                    Bxzx�>  �          @�G�@|���8�ÿ����PQ�C��=@|���=q���H���C�O\                                    Bxzx��  
Z          @�p�@g��C�
���
�q��C��@g��!�������C���                                    Bxzx��  "          @�z�@hQ��:=q��ff���
C���@hQ��33�
=��(�C���                                    Bxzx�0  �          @�=q@Mp���Q���� ��C�  @Mp��k��+��{C��R                                    Bxzx��  "          @�
=@333�E������z�C�#�@333��R������C�C�                                    Bxzx�|  T          @���@R�\�.�R�����Q�C�+�@R�\�p��z���z�C�\                                    Bxzx�"  "          @�Q�@Tz��   ��z���  C�~�@Tz��������C�*=                                    Bxzx��  T          @��@E��^�R�����Q�C�@E��J=q������Q�C�&f                                    Bxzyn  �          @�p�@J=q�����.{��\)C���@J=q�s�
��\)�v�RC��\                                    Bxzy  
�          @��@?\)��=q����ffC�  @?\)�p  ��33���
C�AH                                    Bxzy*�  �          @�ff@<���c33������C��R@<���<(��!���{C��\                                    Bxzy9`  �          @�@:=q�aG���33��Q�C���@:=q�<���Q�����C�N                                    BxzyH  �          @�(�@���l�������(�C��3@���7��P  �&C�)                                    BxzyV�  "          @�\)@%�y����33����C��f@%�O\)�.{��C�K�                                    BxzyeR  
�          @�\)@�R��녿������C���@�R�\���%����
C��)                                    Bxzys�  
�          @���@@���zΰ�
��33C�0�@@�׿���z����C�Ff                                    Bxzy��  �          @�G�@q녿�=q��{���C��@q녿�����
��  C�.                                    Bxzy�D  �          @�Q�@}p��������a��C���@}p���Q�˅���C��q                                    Bxzy��  T          @���@r�\��@  �C�+�@r�\� �׿�33����C�                                      Bxzy��  �          @�
=@W��3�
�J=q�#33C��@W��p�������\)C��                                    Bxzy�6  �          @���@c�
�0  �   ���
C�&f@c�
�\)���
��\)C���                                    Bxzy��  �          @���@p���:=q�B�\�  C�+�@p���.�R���\�F{C�
=                                    Bxzyڂ  �          @�z�@Q��N�R�L�Ϳ��C���@Q��E��z�H�D��C�aH                                    Bxzy�(  T          @�33@����Q�aG��0  C�ff@���qG���\)���HC�#�                                    Bxzy��  
�          @�=q@Mp���?���A�C�S3@Mp��u?���A�p�C��3                                    Bxzzt  "          @���@��@0  @K�B&�HBJ(�@��?�(�@u�BU��B
=                                    Bxzz  
o          @�ff@3�
?�p�@P��B2A���@3�
?8Q�@g�BM=qAfff                                    Bxzz#�  
w          @��@\��=��
@=p�B"Q�?��@\�ͿE�@7
=B�C��{                                    Bxzz2f  "          @�Q�@@  �,��?��
A��RC��)@@  �Fff?p��ADz�C��                                    BxzzA  
�          @��@P  �6ff@�RA�z�C�Z�@P  �\(�?ǮA�(�C��                                    BxzzO�  T          @�\)@<(��Tz�@p�A�{C�Ǯ@<(��s�
?�
=A\  C���                                    Bxzz^X  
�          @��@b�\�9��?��HA�\)C�U�@b�\�U�?��AD(�C�XR                                    Bxzzl�            @�{@1��u?��A|��C��q@1����>L��@C�:�                                    Bxzz{�  �          @��R@Dz��\(�?�A��C��)@Dz��qG�?!G�@��
C���                                    Bxzz�J  �          @�{@K��e?��
AB�RC���@K��o\)���
�aG�C�.                                    Bxzz��            @�{@XQ��b�\>�33@�z�C�ٚ@XQ��`  �&ff����C�f                                    Bxzz��  �          @��R@?\)�vff?+�@��\C��=@?\)�xQ��G����\C�˅                                    Bxzz�<  �          @�
=@,�����\?G�Ap�C�� @,����(��Ǯ��Q�C���                                    Bxzz��  "          @��@7���G�>Ǯ@�{C��3@7��\)�B�\�(�C��H                                    Bxzzӈ  "          @�  @%���ff?&ff@�G�C���@%���
=�
=q�ȣ�C��q                                    Bxzz�.  �          @�{@������>�?ǮC���@�����Ϳ���H  C��3                                    Bxzz��  �          @�p�@���\)?O\)Ap�C�,�@���G��Ǯ���
C�H                                    Bxzz�z  
�          @�{@*=q���
>��H@�{C�h�@*=q���H�+���=qC�}q                                    Bxz{   �          @���@����z�z����C�H�@����녿�\��
=C�@                                     Bxz{�  �          @��\@Q���>��@HQ�C���@Q����H�fff�,  C��                                    Bxz{+l  �          @��
@����{�u�7
=C��@���p���z���G�C�/\                                    Bxz{:  �          @�(�@�\��p�����E�C�b�@�\�n�R�������C���                                    Bxz{H�  �          @�{@G���p��aG��(��C�Ff@G��qG����R����C��                                    Bxz{W^  �          @���@8Q���Q�Q���C���@8Q��h�ÿ����G�C�.                                    Bxz{f  "          @��@333��  ���H�]C�|)@333�aG��  ��z�C�=q                                    Bxz{t�  �          @�  @<(��x�ÿ�G��<  C��f@<(��^{������C�!H                                    Bxz{�P  �          @�  @���\)?�=qA|��C��@���>\)?�\)C��{                                    Bxz{��  �          @�ff?�������?��
A�\)C�|)?�������>��R@h��C��                                    Bxz{��  "          @��?�������?�=qA��
C���?�����G�>�Q�@��C��                                    Bxz{�B  
�          @�p�?��H��{?��RA���C�R?��H��>k�@.�RC��=                                    Bxz{��  "          @��H?h����(�?�  A@(�C��=?h����\)��\)�S33C�n                                    Bxz{̎  �          @��?�����=q?s33A5�C�q?�����p��u�7
=C��H                                    Bxz{�4  T          @�  @   �n{@
=qA�C��)@   ���?��AIC��f                                    Bxz{��  �          @��@33��G�?��
A���C��R@33���?#�
@�33C��                                    Bxz{��  �          @�=q@0���n{?�{A���C�E@0�����\?L��A�C�                                    Bxz|&  �          @�@1G��xQ�?��RA�p�C���@1G�����?c�
A�\C�|)                                    Bxz|�  �          @�{@E�O\)@$z�A�
=C��{@E�s�
?�\)A�Q�C���                                    Bxz|$r  �          @��H@333�h��?��RA�C���@333����?uA0��C�^�                                    Bxz|3  �          @�ff@L���X��@�\A�33C��{@L���s33?���AC
=C��                                    Bxz|A�  �          @���@H�þ�Q�@��\BW�C�@H�ÿ��R@��HBFp�C�XR                                    Bxz|Pd  �          @�p�@R�\�c�
@�ffBK�
C�j=@R�\��(�@tz�B3��C���                                    Bxz|_
  �          @��@P  �fff@�\)BM�\C�Ff@P  ��p�@uB5{C�\)                                    Bxz|m�  �          @��
@Y����33@x��B:��C���@Y����@X��B�C���                                    Bxz||V  �          @�=q@U���
@n�RB2Q�C���@U�,(�@H��B��C���                                    Bxz|��  �          @��R@G��<��@VffB�HC�H�@G��n{@{A�p�C���                                    Bxz|��  �          @��@Q��0��@[�B�RC���@Q��c�
@%A�C�O\                                    Bxz|�H  �          @�p�@N�R�E�@C�
B	�C�&f@N�R�qG�@	��A��C�Ff                                    Bxz|��  �          @�p�@L���G
=@Dz�B	��C��@L���s33@
=qA��
C��                                    Bxz|Ŕ  �          @��R@K��*�H@`  B �RC��R@K��_\)@,��A�  C�+�                                    Bxz|�:  �          @�Q�@L��� ��@n{B)�
C��=@L���X��@<��BG�C���                                    Bxz|��  �          @�\)@?\)�@�=qB@\)C���@?\)�C�
@X��B�C�+�                                    Bxz|�  �          @�\)@Z=q�@o\)B,=qC�E@Z=q�>�R@E�B��C�s3                                    Bxz} ,  �          @�\)@XQ�ٙ�@~{B9p�C���@XQ��*=q@Y��BQ�C���                                    Bxz}�  "          @�
=@Y������@~{B:�C�h�@Y���#�
@[�B��C��                                     Bxz}x  �          @�  @P�׿��
@s�
B:��C�h�@P�����@S33B�
C���                                    Bxz},  �          @��@5�(Q�@C33B33C��
@5�Tz�@�\A�
=C�B�                                    Bxz}:�  �          @�
=@N{�`��?��HA�G�C�E@N{�x��?�G�A4��C���                                    Bxz}Ij  �          @��@Z�H�P  @33A��
C�=q@Z�H�n{?�33Aw�
C�G�                                    Bxz}X  "          @�  @g
=�&ff@1G�A��HC��@g
=�N{@G�A�G�C�(�                                    Bxz}f�  �          @��\@fff�&ff@;�B�C�{@fff�P  @(�A�  C��                                    Bxz}u\  "          @�G�@vff�z�@=p�B�\C���@vff�0  @ffA�
=C�4{                                    Bxz}�  �          @���@c33�33@K�B��C���@c33�AG�@   A�C���                                    Bxz}��  �          @���@e��!�@>�RB�C�h�@e��L(�@  A�G�C�&f                                    Bxz}�N  �          @��@\(��-p�@-p�A�z�C��@\(��S33?���A�(�C��                                    Bxz}��  �          @��@Q��J=q@�A�RC��@Q��j=q?ǮA�G�C��=                                    Bxz}��  p          @��@`���-p�@-p�A�33C�%@`���S33?���A�  C�^�                                    Bxz}�@  �          @�Q�@W���\)@dz�B+p�C��H@W��,��@@  B33C���                                    Bxz}��  �          @�Q�@C�
�@mp�B4\)C���@C�
�<(�@E�B��C��                                    Bxz}�  8          @���@QG���ff@o\)B4p�C��R@QG��*�H@K�B33C�c�                                    Bxz}�2  �          @�Q�@R�\��@|(�BAC�7
@R�\�ff@a�B(\)C���                                    Bxz~�  �          @�G�@^{����@u�B9ffC���@^{��\@[�B!��C��                                    Bxz~~  �          @�\)@c�
��Q�@h��B0p�C��@c�
�33@O\)B�C�\                                    Bxz~%$  �          @�
=@Vff��G�@w
=B?{C��R@Vff��
=@_\)B(\)C��                                    Bxz~3�  
�          @�\)@R�\�=p�@�  BG��C��\@R�\��Q�@l��B4
=C�aH                                    Bxz~Bp  8          @��R@Y���}p�@s33B<
=C��
@Y�����@\(�B&  C�s3                                    Bxz~Q  �          @�p�@aG���p�@^�RB)\)C���@aG���@AG�B��C��f                                    Bxz~_�  "          @�G�@dzῆff@o\)B4z�C��@dz��
=@W�B�HC���                                    Bxz~nb  �          @���@S�
��
=@|(�BA
=C�(�@S�
�
=@b�\B(�C�Ǯ                                    Bxz~}  "          @�G�@J�H�8Q�@��BP  C���@J�H�ٙ�@w
=B<(�C���                                    Bxz~��  �          @���@N�R��{@�(�BO
=C��R@N�R��=q@z�HBA�C��3                                    Bxz~�T  �          @�=q@_\)�L��@{�B>�RC���@_\)��(�@g�B+�
C��H                                    Bxz~��  
�          @��@�=q�s33@R�\B��C�o\@�=q���H@>{B��C���                                    Bxz~��  
�          @�33@}p���\)@L��BffC�޸@}p��@.{A�
=C���                                    Bxz~�F  �          @��@��H��@@  B��C��=@��H��@ ��A���C�!H                                    Bxz~��  �          @��H@�Q��
=@:�HB��C�(�@�Q��$z�@�A�C��                                    Bxz~�  �          @�{@����{@>{B{C��\@���!G�@(�AԸRC�k�                                    Bxz~�8  �          @�{@�  ���R@/\)A�RC�u�@�  �%�@(�A�ffC�XR                                    Bxz �  �          @��@���ff@333A��C�z�@���,��@{A�=qC�XR                                    Bxz�  �          @��@�\)�G�@\)AۮC��@�\)�2�\?��A�G�C�E                                    Bxz*  T          @��@������@��A��C�h�@����7�?У�A��
C�'�                                    Bxz,�  �          @�(�@����   @�A�p�C�xR@����>�R?�A��C�*=                                    Bxz;v  �          @�33@���   @�A�
=C��)@���:�H?�
=Axz�C���                                    BxzJ  �          @��H@�G�� ��?��HA�C���@�G��9��?��Ac\)C���                                    BxzX�  "          @��H@�
=�(�@��A�ffC�J=@�
=�,(�?���A�  C��q                                    Bxzgh  
�          @�=q@�ff�{@��AՅC�{@�ff�-p�?�ffA��\C��\                                    Bxzv  �          @�33@�  ��R@
=A�C�#�@�  �.{?��
A�G�C���                                    Bxz��  �          @��@�����H@  A�33C�=q@����7�?У�A�=qC�                                    Bxz�Z  �          @���@���!G�?.{@��C���@���'
==�\)?L��C�)                                    Bxz�   �          @�=q@�33��R?h��A��C��R@�33�'�>��R@U�C�*=                                    Bxz��  T          @���@�z��33?�  A�\)C��@�z��?z�HA*�RC���                                    Bxz�L  �          @�G�@�z��33?�33Aw�C�\@�z��z�?c�
A  C��{                                    Bxz��  �          @���@����\?���AR{C�^�@�����?333@�C�@                                     Bxzܘ  �          @��@�z��
=?}p�A+
=C�>�@�z��ff?�@��C�\)                                    Bxz�>  �          @��@���
�H?�
=AL(�C��3@���Q�?&ff@�  C��f                                    Bxz��  T          @���@�G���
?n{A!C�b�@�G��{>�(�@�ffC��R                                    Bxz��  T          @���@�  ����?8Q�@��C�%@�  ��(�>�{@hQ�C���                                    Bxz�0  �          @�G�@����Q�?Q�A
=C��@����G�>��R@X��C�J=                                    Bxz�%�  �          @�G�@��
����?(��@�
=C��@��
�33>L��@p�C���                                    Bxz�4|  T          @�33@�(���(�?�{A�{C���@�(����?�p�AUG�C���                                    Bxz�C"  �          @��@�  ��(�?�\)A��\C�h�@�  ����?�p�A��
C�W
                                    Bxz�Q�  "          @��H@�p���33@
=qA�G�C��f@�p�����?��A��RC�,�                                    Bxz�`n  
�          @��H@�Q����?�=qA��HC�� @�Q��Q�?�
=Ay�C��                                    Bxz�o  T          @�33@�ff���
@�
A��C���@�ff���H?�z�A��RC���                                    Bxz�}�  �          @�33@�G���?\A�=qC���@�G��ff?���A9��C�#�                                    Bxz��`  "          @��H@��ÿ��R?��Ac�C��{@�����R?Tz�A�C�|)                                    Bxz��  �          @�33@��ÿ�(�?��Ac\)C��3@����p�?W
=AQ�C���                                    Bxz���  �          @��
@�녿�p�?�\)A�
=C��@���33?���AN=qC�u�                                    Bxz��R  T          @�@��H�G�?���AlQ�C���@��H��?fffA  C�^�                                    Bxz���  "          @�z�@�G��33?�\)Alz�C�h�@�G��33?aG�A�HC�&f                                    Bxz�՞  "          @�{@�����?˅A��C��q@���(�?���A6=qC�e                                    Bxz��D  �          @�Q�@�\)�  ?�
=A�(�C�G�@�\)�$z�?��A>�HC�                                    Bxz���  �          @���@��ff?�p�A�Q�C���@��+�?�AC�C�)                                    Bxz��  �          @�G�@�  �?�z�A�33C��H@�  �)��?���A733C�k�                                    Bxz�6  �          @��R@�{���?�Q�At��C�s3@�{�*=q?aG�A�C�:�                                    Bxz��  �          @�(�@�����?���A�p�C�� @���/\)?}p�A"{C�/\                                    Bxz�-�  �          @�{@�p���?�{Ag�C�@ @�p��*�H?J=qA�C��                                    Bxz�<(  �          @��R@���� ��?���Aep�C���@����0  ?E�AG�C��{                                    Bxz�J�  �          @���@�\)�%�?�33A@(�C��@�\)�1G�?��@���C�Ф                                    Bxz�Yt  �          @�33@�=q�5�?��A�{C�)@�=q�Fff?c�
A{C���                                    Bxz�h  �          @�{@��R�5�?���AmG�C�z�@��R�E�?L��AC�c�                                    Bxz�v�  �          @��
@�\)�1�?}p�A)p�C�{@�\)�;�>�Q�@s�
C�h�                                    Bxz��f  "          @�33@��1�?�=qA:�HC��\@��<��>�ff@�33C�*=                                    Bxz��  �          @�(�@��
�=p�?�(�AH(�C���@��
�J=q?��@��\C���                                    Bxz���  �          @���@��\�C33?��AR�RC�0�@��\�P  ?��@�=qC�O\                                    Bxz��X  �          @�p�@����C33?�Ahz�C�\@����R�\?:�H@�(�C��                                    Bxz���  �          @�@�{�E�?��HA�{C���@�{�XQ�?�  A"�RC�`                                     Bxz�Τ  T          @��R@��Mp�?�\)A�(�C��@��_\)?c�
A(�C��f                                    Bxz��J  �          @�33@����O\)?�{A��C�e@����`��?^�RA�C�G�                                    Bxz���  T          @��\@�33�K�?��
A]p�C�R@�33�XQ�?�@��HC�=q                                    Bxz���  
�          @�  @��S33?��Aj�\C��)@��aG�?(��@ڏ\C��3                                    Bxz�	<  �          @��@���I��?z�HA'�C��{@���Q�>�\)@@  C��                                    Bxz��  
�          @��@��Mp�?��HAG
=C��@��Y��>��H@��C�@                                     Bxz�&�  T          @�G�@����C�
?���AfffC��
@����R�\?0��@�ffC���                                    Bxz�5.  �          @���@��C33?��AZ�\C���@��P��?!G�@У�C�Ф                                    Bxz�C�  �          @���@���A�?��\A)�C��)@���K�>�33@h��C�Z�                                    Bxz�Rz  T          @���@��R�=p�?���Aa�C�5�@��R�K�?0��@�C�B�                                    Bxz�a   �          @��@����@��>�\)@>�RC���@����@  ��Q��z�HC��
                                    Bxz�o�  
�          @��\@�{�7��333��33C��3@�{�)������`��C���                                    Bxz�~l  �          @��H@��\�C�
�(��ҏ\C�b�@��\�7
=���R�W33C�Ff                                    Bxz��  
�          @���@���Mp������C�*=@���B�\��33�H��C���                                    Bxz���  �          @���@���QG����H���\C�� @���E��z��L(�C���                                    Bxz��^  �          @�{@��
�c33�#�
���
C���@��
�]p��@  ��C���                                    Bxz��  T          @�  @�p��a�?   @�
=C��H@�p��c�
��  �'
=C��f                                    Bxz�Ǫ  
Z          @�z�@}p��e>�\)@B�\C��@}p��dz������C�                                      Bxz��P  �          @�@�Q��h��>�{@dz�C��@�Q��hQ��(�����C��                                    Bxz���  �          @��@�
=�e�?
=q@���C��)@�
=�g
=�aG��  C���                                    Bxz��  T          @���@�\)�^�R?=p�@�
=C�H�@�\)�c�
    =#�
C��R                                    Bxz�B  T          @��@���j�H?   @�Q�C��@���l(���=q�333C��R                                    Bxz��  "          @��@�  �n�R>��@�=qC�xR@�  �o\)��p��w
=C�s3                                    Bxz��  T          @���@��R�`��?#�
@�{C��@��R�dz���Ϳ�  C���                                    Bxz�.4  T          @�=q@~�R�a�>���@fffC�9�@~�R�aG��������C�@                                     Bxz�<�  T          @���@����X��>�z�@K�C���@����XQ�����
=C��                                    Bxz�K�  �          @�Q�@�  �Z=q>�=q@;�C��H@�  �X�þ�G�����C��{                                    Bxz�Z&  �          @�
=@~{�XQ�>8Q�?�p�C��\@~{�U����C���                                    Bxz�h�  
�          @�\)@�p��I����  �,(�C�w
@�p��A녿c�
�  C��)                                    Bxz�wr  
D          @��\@����_\)�����c�
C���@����Vff����3�C��                                    Bxz��  �          @�\)@p���hQ�����C���@p���a녿\(��\)C�ff                                    Bxz���  �          @�  @fff�r�\>�z�@Mp�C���@fff�qG����H���\C���                                    Bxz��d  
Z          @�G�@Y���~{?Y��A33C�J=@Y�����<��
>aG�C���                                    Bxz��
  
�          @�ff@e�l(�?&ff@�\)C�)@e�p  ��G���p�C��                                    Bxz���  "          @�
=@`���tz�?�@�(�C�H�@`���vff��  �1G�C�.                                    Bxz��V  �          @�
=@g��n�R>��@�Q�C�\@g��o\)��Q��z=qC�
=                                    Bxz���  �          @��R@^{�u?\)@��
C�
=@^{�w��k��$z�C��                                    Bxz��  
Z          @��\@\���mp�?�
=A�z�C�t{@\���~�R?c�
A�C�z�                                    Bxz��H  �          @�\)@c�
�c33@(�A�(�C���@c�
�z=q?�
=Aqp�C�"�                                    Bxz�	�  T          @���@dz��dz�@�\A���C�~�@dz��}p�?\A~�RC�                                    Bxz��  �          @��@b�\�e@A���C�L�@b�\�\)?���A�
=C��                                    Bxz�':  �          @��@hQ��c33@G�A��\C��=@hQ��|(�?�G�A{�C�S3                                    Bxz�5�  �          @��\@u��c33?�(�A�\)C��3@u��w�?�(�AIC�T{                                    Bxz�D�  �          @��@qG��`��@�A��C���@qG��w
=?�=qA]C�+�                                    Bxz�S,  "          @��\@u�U@�
A�33C�}q@u�n�R?˅A��\C��                                    Bxz�a�  
�          @�(�@e�~�R?�p�A��RC��@e���?fffA�\C�"�                                    Bxz�px  �          @�33@fff�~{?У�A��C�
@fff���R?O\)Az�C�<)                                    Bxz�  �          @��@a����?�Q�An�\C�Q�@a����?
=@��HC��H                                    Bxz���  T          @�33@XQ�����?��A��\C���@XQ����
?0��@�=qC���                                    Bxz��j  �          @���@_\)��=q?���A\��C�H�@_\)��  >�@�  C���                                    Bxz��  
�          @�@W
=�u?�A�z�C��H@W
=���
?��\A,��C���                                    Bxz���  T          @�{@R�\�n{@p�A�  C��H@R�\���H?�
=At(�C�o\                                    Bxz��\  �          @�z�@[��p  ?��
A�C�:�@[��~�R?B�\A�HC�c�                                    Bxz��  �          @�z�@g��mp�?�Au�C�*=@g��z�H?(��@ᙚC�b�                                    Bxz��  �          @�{@g��i��?�(�A��C�]q@g��z�H?uA#�C�Z�                                    Bxz��N  T          @�ff@|(��i��?fffA��C���@|(��p��>.{?�  C�1�                                    Bxz��  �          @��R@z�H�n{?:�H@�p�C�7
@z�H�r�\���
�L��C��3                                    Bxz��  "          @�ff@y���j�H?c�
A33C�Z�@y���qG�>��?�{C���                                    Bxz� @  �          @��R@�Q��j�H?�@��C�� @�Q��mp��L���z�C���                                    Bxz�.�  �          @�{@���_\)>#�
?��C��@���\�Ϳ����C�/\                                    Bxz�=�  "          @��R@����S�
?��A<(�C�^�@����]p�>�
=@�Q�C���                                    Bxz�L2  �          @��@p���Dz�@��A��
C�]q@p���^�R?�  A�z�C��R                                    Bxz�Z�  �          @��
@a��9��@4z�A�\)C�U�@a��X��@��A�p�C�R                                    Bxz�i~  �          @�p�@a��Fff@+�A��
C�c�@a��c�
@�A�{C�e                                    Bxz�x$  �          @�\)@p���Q�@  A�(�C�t{@p���j=q?���A��C��                                    Bxz���  �          @��@w��[�?�A�z�C�5�@w��n{?�{A;
=C��                                    Bxz��p  T          @�  @{��dz�?�
=Ap��C���@{��q�?333@�C�
=                                    Bxz��  T          @�Q�@\)�\(�?���A�G�C���@\)�k�?fffA33C���                                    Bxz���  
Z          @�@{��Mp�?�33A�(�C�]q@{��aG�?��RAS�
C��                                    Bxz��b  T          @��@y���K�?��RA�ffC�h�@y���`  ?�=qAdQ�C��                                    Bxz��  �          @���@tz��E@�RA�33C�� @tz��]p�?˅A���C��                                    Bxz�ޮ  �          @�p�@z�H�Fff@33A�{C��
@z�H�\(�?�z�AqC�b�                                    Bxz��T  "          @�p�@xQ��G
=@
�HA�33C��@xQ��^{?��
A��C�
                                    Bxz���  "          @�@{��J=q@   A�G�C���@{��_\)?�{Ag�C�1�                                    Bxz�
�  �          @�{@�  �S33?У�A�p�C�<)@�  �c33?uA#
=C�0�                                    Bxz�F  �          @�@tz��S�
?�
=A�G�C���@tz��g�?�G�AW33C�J=                                    Bxz�'�  
�          @�  @x���I��@�A�C�y�@x���`  ?�(�A|  C��q                                    Bxz�6�  �          @��@i���U@�A�(�C��q@i���n{?���A�ffC�7
                                    Bxz�E8  
�          @�p�@l(��c�
?޸RA�{C�  @l(��u�?��\A.�RC���                                    Bxz�S�  "          @�p�@s33�aG�?ǮA��
C���@s33�p��?Y��AG�C��\                                    Bxz�b�  "          @��R@\)�^�R?��AZ�RC�|)@\)�j=q?��@��C�                                    Bxz�q*  �          @�p�@����J�H?�As�C�L�@����XQ�?J=qA{C�e                                    Bxz��  �          @��@�  �C�
?�\)Aj{C�!H@�  �P��?B�\A ��C�=q                                    Bxz��v  �          @�ff@����(Q�?�p�A}�C��@����7
=?s33A!��C��R                                    Bxz��  �          @�\)@�p���?��RA}��C�9�@�p��+�?�  A(��C�
                                    Bxz���  �          @���@�z��?�\)A��\C�� @�z��)��?��Ai��C�                                      Bxz��h  �          @���@��(��?���A��
C��R@��=p�?�z�Am��C��                                    Bxz��  �          @���@��%�@�A��RC��f@��;�?�ffA�Q�C�E                                    Bxz�״  �          @�G�@�ff�"�\@�A��C�q@�ff�8��?ǮA�G�C�y�                                    Bxz��Z  �          @�Q�@���!�@z�A��C��\@���:�H?�ffA�=qC��                                    Bxz��   �          @���@����(Q�@�
A�G�C�0�@����AG�?�G�A���C�^�                                    Bxz��  
�          @�\)@�(��6ff@��A��C��R@�(��Mp�?�\)A��C��                                    Bxz�L  �          @��@�����@33Aƣ�C���@���2�\?�A��
C��                                    Bxz� �  �          @��@�G��'
=@�RA�\)C�\)@�G��>�R?�Q�A���C���                                    Bxz�/�  �          @�{@��#33@
=A��C�U�@��<��?�A�  C�j=                                    Bxz�>>  
�          @�p�@�p��\)@��A�ffC���@�p��9��?��A�
=C��q                                    Bxz�L�  T          @�p�@�ff�z�@,��A�{C���@�ff�"�\@G�A�(�C�g�                                    Bxz�[�  �          @�=q@H���tz�?���A�{C���@H����33?���A;�C���                                    Bxz�j0  T          @���@L���g�@�A�=qC��)@L���|(�?��Ag�C���                                    Bxz�x�  T          @��@QG��g�@�A��RC�@QG��{�?��Ae�C��q                                    Bxz��|  �          @��@.�R����@Q�A��C�3@.�R��33?��Ai��C��                                    Bxz��"  T          @��@4z��|��@��A�(�C��)@4z�����?�{Al��C���                                    Bxz���  �          @�G�@(�����H@33A��\C�n@(�����?��RAYC�|)                                    Bxz��n  �          @��H@����
=@��A��
C�@����G�?��AdQ�C�%                                    Bxz��  T          @�=q@p����@�
A�ffC�\@p����?�p�AU��C�1�                                    Bxz�к  �          @��H@ff���\@�\A�p�C�B�@ff��z�?�Q�AMp�C�t{                                    Bxz��`  �          @�G�@����R?�p�A��C�B�@���  ?�{AAp�C���                                    Bxz��  �          @�=q@���@A�G�C���@����?�p�AU�C��R                                    Bxz���  �          @��@
=����@�A�=qC��\@
=��
=?��HAS�C��                                    Bxz�R  �          @��H@���{@ffA��C��=@���  ?��RAV�\C���                                    Bxz��  T          @��
@���\)@
=A�p�C�p�@���G�?��RAUG�C���                                    Bxz�(�  �          @��
@
=����?��HA�G�C�!H@
=��{?���A=p�C�b�                                    Bxz�7D  �          @��@�\��p�@
=A���C���@�\���?��RAT��C��                                    Bxz�E�  �          @���@Q����@�
A˙�C��@Q���
=?���A{33C�&f                                    Bxz�T�  �          @�z�@z���  @�A�33C�\)@z����?��RAT��C���                                    Bxz�c6  T          @���?�(�����?Y��A{C�9�?�(�����L�Ϳz�C��                                    Bxz�q�  �          @��H?�ff����?��\A0z�C�3?�ff��Q�=�Q�?k�C��H                                    Bxz���  �          @��?�����?�\A�ffC��)?������?W
=Az�C�t{                                    Bxz��(  T          @�z�?�p���33?�p�A���C���?�p���z�?��A:�HC�                                    Bxz���  �          @�p�@	����=q?�Q�A�  C���@	����33?�ffA3�C��R                                    Bxz��t  
�          @�\)@)�����?��RA�ffC��f@)������?��A@��C���                                    Bxz��  "          @�=q@��G�@ffA�z�C�L�@���?��\A\��C�u�                                    Bxz���            @�=q@p���
=@
=A�33C�)@p���G�?��A`��C�7
                                    Bxz��f  �          @��@#33����?��RA��RC�ff@#33��=q?�z�AH��C���                                    Bxz��  �          @�z�@����{?���A��C�(�@�����R?}p�A*{C�y�                                    Bxz���  �          @�=q?�33��=q?�\)A��C���?�33���\?p��A$Q�C�Z�                                    Bxz�X  �          @��H?�������?�p�A���C��H?������?��A7�
C�t{                                    Bxz��  �          @�33?�������?�=qA�{C�k�?������?fffA�C��                                    Bxz�!�  �          @�G�?�33���?��A��
C�:�?�33���?fffA��C��3                                    Bxz�0J  �          @�G�@����=q@�A�Q�C�,�@������?�=qAi�C�\)                                    Bxz�>�  �          @�=q?��R��G�?�33A��
C��
?��R��=q?��
A2{C�<)                                    Bxz�M�  �          @�  ?}p����R@�A��
C���?}p���Q�?�z�AC�C�XR                                    Bxz�\<  "          @�ff?z�H��\)?�A��C���?z�H��  ?uA#33C�N                                    Bxz�j�  �          @�\)?����z�@33A���C�� ?����?�{A;�
C�
                                    Bxz�y�  �          @�ff?�ff��ff?���A���C�^�?�ff���R?^�RA�
C�f                                    Bxz��.  �          @�  ?p������?��HA���C�H�?p����=q?}p�A%G�C�                                    Bxz���  �          @��R?\(����R@
=A�ffC��f?\(���Q�?�z�AD  C���                                    Bxz��z  �          @��?+����
?�{A�z�C��R?+����
?aG�A\)C���                                    Bxz��   �          @�ff>8Q���ff?�=qA���C�H>8Q����?
=@�C��R                                    Bxz���  �          @�>�=q��
=?�z�AqG�C�|)>�=q��z�>�
=@��C�o\                                    Bxz��l  �          @��?������R?�z�A�
=C��?�����{?5@��C�Ф                                    Bxz��  �          @��H?�ff���R?�z�A�G�C��{?�ff��?5@�{C��{                                    Bxz��  �          @��\?����(�?�(�A��C�b�?�����?J=qAz�C�\                                    Bxz��^  �          @��\?�{���
?�z�A�33C���?�{���H?:�H@��HC�~�                                    Bxz�  �          @�G�?�Q���p�?��
A�C��=?�Q���p�?aG�A{C�s3                                    Bxz��  �          @�{?�{����?�A��
C���?�{��(�?G�A	��C�!H                                    Bxz�)P  �          @��?�\)��33?�{A�33C��\?�\)���
?z�HA,��C�0�                                    Bxz�7�  �          @�Q�?aG����R?�\)As�C�
=?aG���(�>�(�@�{C��                                     Bxz�F�  �          @���?p�����R?��
A���C�^�?p�����?z�@�=qC�+�                                    Bxz�UB  �          @��?L����
=?�Q�A�  C���?L������?   @�{C�h�                                    Bxz�c�  �          @���?^�R��ff?���A���C���?^�R���?�R@�G�C���                                    Bxz�r�  �          @�Q�?@  ��p�?��HA��
C�U�?@  ����?E�AffC�%                                    Bxz��4  �          @�\)?p�����H?�(�A�  C���?p����=q?J=qA�
C�C�                                    Bxz���  �          @�Q�?�33����?�{A�{C��=?�33��G�?p��A%��C�q�                                    Bxz���  �          @���?Ǯ��(�?��RA��C�U�?Ǯ��p�?��A?�C��3                                    Bxz��&  �          @�Q�?�z���
=@��A�C�33?�z�����?�=qAjffC��                                    Bxz���  �          @���?\��p�?�p�A�C�?\���R?�=qA<(�C���                                    Bxz��r  �          @��\?�Q����
@�\A�(�C�\?�Q���p�?�33AF�HC�~�                                    Bxz��  �          @���?�=q���?�{A���C�Ф?�=q���
?xQ�A)C�G�                                    Bxz��  
�          @�\)?�G���\)@�\A��
C��q?�G�����?�AO�C��                                    Bxz��d  �          @��R@   ����@��A��
C���@   ���?�\)As\)C��q                                    Bxz�
  �          @�{@   ���@z�A���C�t{@   ���
?��RA]�C���                                    Bxz��  �          @��R@p���Q�@p�A��C��@p���33?�A~ffC��                                     Bxz�"V  �          @�
=@����33@33AхC�o\@�����R?�  A��C�q�                                    Bxz�0�  �          @��@����H?�Q�A�Q�C�T{@���(�?���AB=qC��)                                    Bxz�?�  �          @�{@%����@ ��A��C�aH@%���\?�(�AY�C�q�                                    Bxz�NH  �          @��@&ff����@A���C�s3@&ff���H?�ffAg\)C�xR                                    Bxz�\�  �          @�ff@?\)�dz�@�Aԏ\C���@?\)�|(�?�\)A�G�C��
                                    Bxz�k�  �          @��R@5��c33@#33A�\C�G�@5��}p�?�A��C���                                    Bxz�z:  T          @�\)@J�H�N�R@#�
A�{C�8R@J�H�i��?�z�A��RC�y�                                    Bxz���  �          @��@K��@��@5B  C�N@K��_\)@�RA�z�C�1�                                    Bxz���  �          @��@S33�1�@<��B	Q�C���@S33�Q�@Q�A�ffC��)                                    Bxz��,  T          @��R@K��9��@<(�B	p�C��3@K��Y��@ffA�=qC��                                    Bxz���  �          @��R@C�
�C33@:�HBffC��=@C�
�b�\@33Aљ�C�ff                                    Bxz��x  �          @�{@J�H�A�@0��B �\C�&f@J�H�_\)@��A�\)C��                                    Bxz��  �          @��@HQ��=p�@>{B
z�C�G�@HQ��^{@�A��C��                                    Bxz���  �          @�\)@<���@  @FffB�C�@ @<���a�@�RA�
=C���                                    Bxz��j  �          @�ff@3�
�E�@G
=B�C�'�@3�
�g
=@�RA�C��                                    Bxz��  T          @�{@4z��C33@G�B�RC�` @4z��e�@\)A噚C�)                                    Bxz��  �          @��R@0���N�R@@��BG�C�AH@0���n�R@ffA���C�9�                                    Bxz�\  �          @�p�@1��P��@8Q�BC�7
@1��o\)@{AˮC�J=                                    Bxz�*  �          @��@!G��~{?�Q�A�
=C�:�@!G���Q�?�z�AR�\C�Q�                                    Bxz�8�  T          @�(�@   ����?���Ao33C�&f@   ��ff?   @���C��=                                    Bxz�GN  �          @��
@(Q��|(�?��A�z�C�� @(Q���
=?�\)AK33C���                                    Bxz�U�  �          @�@%�i��@%�A�C��@%��=q?���A�C�<)                                    Bxz�d�  �          @�@���{�@z�A�z�C��3@������?��
A��C��{                                    Bxz�s@  �          @�p�@#�
�r�\@��A�G�C���@#�
��p�?��A�(�C��q                                    Bxz���  �          @�p�@%�p  @�A�ffC�N@%��z�?�
=A��C�H                                    Bxz���  �          @�{@O\)�p�@N{Bz�C�k�@O\)�AG�@,��A��HC���                                    Bxz��2  �          @�{@P  �$z�@G
=B��C��q@P  �G
=@$z�A�{C�(�                                    Bxz���  �          @�ff@Tz��!G�@G
=B��C�ff@Tz��C�
@$z�A홚C��=                                    Bxz��~  �          @�ff@P  �+�@A�B  C�9�@P  �Mp�@{A�p�C��                                    Bxz��$  �          @���@Y���(Q�@5�B�\C�(�@Y���G�@�\A���C�                                    Bxz���  �          @�ff@Z�H�4z�@.{A�{C�H�@Z�H�Q�@��A�  C��                                    Bxz��p  �          @��R@\(��:�H@&ffA��C��@\(��W
=@   A��HC��                                    Bxz��  �          @��R@Z=q�>{@%�A�C�y�@Z=q�Z=q?�(�A��
C��f                                    Bxz��  �          @�ff@Vff�@��@&ffA�C�f@Vff�\��?�p�A��RC��                                    Bxz�b  �          @�
=@U��G�@!�A�ffC�c�@U��c33?�33A�Q�C���                                    Bxz�#  �          @�z�@Y���<(�@(�A�{C��
@Y���Vff?�A��\C��R                                    Bxz�1�  �          @��@dz��#33@%A���C�8R@dz��@  @�
A��RC��)                                    Bxz�@T  �          @�z�@dz��0  @p�A��C�/\@dz��J�H?��A�
=C�1�                                    Bxz�N�  �          @�z�@g
=�/\)@�A�  C�^�@g
=�J=q?�\)A�=qC�ff                                    Bxz�]�  �          @�z�@j=q�(��@p�A�
=C��@j=q�C�
?�z�A��HC��                                    Bxz�lF  �          @���@k��$z�@!�A���C��@k��@  ?��RA�C�ff                                    Bxz�z�  �          @���@l���!�@!�A�G�C��3@l���>{@   A��RC��                                    Bxz���  �          @�(�@qG���R@�A�RC�Q�@qG��9��?�z�A�\)C�5�                                    Bxz��8  �          @��R@q��#�
@   A�p�C��
@q��?\)?��HA��C��R                                    Bxz���  �          @�ff@g
=�'�@,(�A��\C�f@g
=�E�@��A�C���                                    Bxz���  �          @�{@o\)�   @%�A�
=C��@o\)�<��@�
A��HC���                                    Bxz��*  �          @�\)@u�!G�@ ��A�
=C�S3@u�=p�?�p�A�p�C�,�                                    Bxz���  �          @��R@u�(��@z�A��
C�@u�A�?�\A��C��R                                    Bxz��v  �          @�{@z=q� ��@�
AӅC���@z=q�:=q?��
A��HC���                                    Bxz��  �          @���@E�R�\?��RA�{C���@E�g
=?���Aw�
C�O\                                    Bxz���  �          @�p�@)����z�?���A�C�K�@)�����
?E�AQ�C���                                    Bxz�h  �          @�p�@1G�����?���A��RC�(�@1G���G�?J=qA��C�l�                                    Bxz�  �          @��
@8������?�
=AUp�C��
@8����>\@�
=C�W
                                    Bxz�*�  �          @��@0  ���?�  Ab�RC���@0  ����>�(�@��\C�`                                     Bxz�9Z  �          @�G�@%����?�\)AN=qC���@%����>�z�@W
=C��\                                    Bxz�H   �          @���@  ����?��RAeC�޸@  ��{>\@��C�p�                                    Bxz�V�  �          @��\@������?�ffA?\)C��R@����p�>B�\@(�C��                                     Bxz�eL  �          @��@!G���G�=�Q�?��C�7
@!G���\)�=p��	��C�ff                                    Bxz�s�  
�          @���@!G�����?(��@�G�C�B�@!G���=q�B�\�	��C�                                      Bxz���  �          @���@2�\���\?}p�A5C�0�@2�\��ff>8Q�@�C���                                    Bxz��>  �          @��H@*�H����?��AP(�C�aH@*�H��G�>��R@_\)C��                                    Bxz���  �          @�(�@�
��{?�Q�A�{C�p�@�
��{?W
=A�C��                                     Bxz���  �          @��@�
����?�p�A�{C�޸@�
���?�33AQ�C���                                    Bxz��0  �          @�z�@%�w�@�
A���C�� @%��ff?�G�Ae��C�Ф                                    Bxz���  �          @�{@ ���QG�@\��B({C���@ ���xQ�@0  B ��C���                                    Bxz��|  �          @�{@ff�g
=@C�
B�
C�\@ff����@�\A��C�j=                                    Bxz��"  �          @��R@{�Y��@P��B�
C��\@{�~�R@!G�A��C���                                    Bxz���  �          @�
=@(��`  @>�RB�C�s3@(�����@�RA��HC��                                    Bxz�n  �          @��R@G��i��@:�HBC��=@G����@��A���C�N                                    Bxz�  �          @��H@��o\)@��A�(�C�z�@�����?�z�A�z�C�&f                                    Bxz�#�  �          @�p�@#33�c�
@.{A�\)C�˅@#33����?�(�A��C�'�                                    Bxz�2`  �          @�p�@(���dz�@)��A��HC�7
@(������?��A�C���                                    Bxz�A  �          @�p�@%�qG�@��A�Q�C�AH@%��p�?˅A��C��\                                    Bxz�O�  �          @���@/\)�q�@	��A�{C��\@/\)��z�?�{AuG�C���                                    Bxz�^R  �          @��@%�r�\@Aأ�C�+�@%��?�ffA��C��                                    Bxz�l�  �          @��@:=q�r�\?�A��C���@:=q���H?���AK33C��R                                    Bxz�{�  �          @���@,���tz�@Q�AÅC��
@,����p�?���AnffC�o\                                    Bxz��D  �          @���@3�
�o\)@�A�Q�C�y�@3�
��33?��Az�RC�9�                                    Bxz���  �          @�(�@:�H�e�@p�A�Q�C�� @:�H�|��?���A�\)C�B�                                    Bxz���  �          @�=q@/\)�l��@�A�p�C�E@/\)����?�=qAs�
C�                                    Bxz��6  �          @��@@���\(�@�A��
C���@@���u�?��A�z�C��                                    Bxz���  �          @��@8Q��^�R@\)A���C��3@8Q��z=q?޸RA��
C�5�                                    Bxz�ӂ  �          @�(�@.�R�l(�@G�AҸRC�E@.�R��=q?��RA���C��                                    Bxz��(  �          @�z�@(Q��n�R@�A�z�C���@(Q���(�?��
A�33C�Ff                                    Bxz���  T          @�z�@/\)�c�
@\)A�{C��=@/\)�\)?�(�A���C�=q                                    Bxz��t  �          @���@7��aG�@=qA߮C���@7��{�?�33A�C�\                                    Bxz�  �          @�p�@@  �^{@�A�
=C�o\@@  �w�?�\)A���C�޸                                    Bxz��  �          @���@=p��\��@�HA�{C�T{@=p��w�?�A�33C���                                    Bxz�+f  �          @��@E��P��@#33A���C���@E��mp�?�=qA��\C���                                    Bxz�:  �          @�(�@HQ��XQ�@  A�  C�n@HQ��p��?�G�A�\)C���                                    Bxz�H�  �          @�33@G
=�S�
@�A�
=C���@G
=�l��?�ffA��\C�                                      Bxz�WX  �          @���@QG��%�@7
=B	�HC��H@QG��G
=@G�A��C�<)                                    Bxz�e�  �          @�  @J=q�Fff@�AٮC��f@J=q�`��?���A�p�C�
=                                    Bxz�t�  �          @���@I���K�@�A��C�]q@I���e?У�A���C���                                    Bxz��J  �          @�
=@J=q�@��@
=A���C�1�@J=q�[�?�Q�A�G�C�T{                                    Bxz���  �          @�
=@<���?\)@)��B�C�XR@<���^{?�p�A�=qC�7
                                    Bxz���  �          @�\)@5��8��@5BffC�*=@5��Z�H@(�A�  C���                                    Bxz��<  �          @�ff@C33�7
=@(��B
=C�b�@C33�Vff?��RA�{C�'�                                    Bxz���  �          @�{@>{�5@/\)BC�"�@>{�U@AȸRC���                                    Bxz�̈  �          @�ff@G��E�@�A�(�C���@G��^�R?���A���C��3                                    Bxz��.  �          @�{@L(��Dz�@�A�G�C��@L(��]p�?��RA���C�]q                                    Bxz���  �          @�ff@Mp��HQ�@z�A�Q�C���@Mp��_\)?���A��C�Q�                                    Bxz��z  �          @�{@I���N{@�A�=qC�8R@I���dz�?���Aw�C��)                                    Bxz�   �          @��@G
=�L��@�\AĸRC�)@G
=�c�
?�=qA{�
C���                                    Bxz��  �          @���@>�R�N�R@
=qAхC�]q@>�R�g
=?�Q�A���C���                                    Bxz�$l  �          @��
@E��Mp�@   A�=qC��@E��c33?��Aup�C�o\                                    Bxz�3  �          @�z�@G
=�Mp�?�p�A�
=C�3@G
=�c33?�G�Ao�C��                                     Bxz�A�  �          @��@=p��Tz�@33AŮC�޸@=p��j�H?�ffAw\)C�n                                    Bxz�P^  �          @�\)@G��Vff?�(�A�Q�C�}q@G��l(�?�(�Ac33C��                                    Bxz�_  �          @�
=@Fff�U�?�p�A��\C�� @Fff�j�H?��RAg\)C��                                    Bxz�m�  �          @��R@E�S33@�A�33C���@E�i��?��Ap��C�q                                    Bxz�|P  �          @�{@L���QG�?�{A��RC�9�@L���e�?���AR�RC��                                    Bxz���  �          @��@?\)�L(�@  A�p�C���@?\)�fff?\A��C��)                                    Bxz���  �          @�(�@5�QG�@p�A�Q�C�|)@5�j=q?�(�A�z�C��                                    Bxz��B  �          @��H@ ���g
=@ ��A��HC�c�@ ���}p�?���Af{C�/\                                    Bxz���  �          @��\@%�c�
?�(�A��\C��@%�y��?�z�A_�C�Ф                                    Bxz�Ŏ  �          @��H@'��dz�?���A��\C�q@'��z=q?�33A[�C���                                    Bxz��4  �          @��@(���c�
?��RA��C�P�@(���y��?�Q�Ab=qC��                                    Bxz���  �          @�33@*�H�a�?���A�{C��@*�H�w�?�33A[�C�Q�                                    Bxz��  �          @���@*�H�`  ?�{A�33C��{@*�H�s�
?���ANffC��                                    Bxz� &  �          @��@.{�\(�?��A�C�.@.{�o\)?�G�AE�C��                                    Bxz��  �          @�
=@)���_\)?޸RA�G�C��3@)���r�\?s33A9p�C�z�                                    Bxz�r  
�          @�
=@*=q�^�R?�p�A�Q�C���@*=q�q�?p��A7�C��{                                    Bxz�,  �          @��@%��Y��@�
A�{C���@%��qG�?��
AyC�/\                                    Bxz�:�  �          @�\)@$z��^�R?�(�AîC�8R@$z��tz�?�Ac�C��3                                    Bxz�Id  �          @��@&ff�`��?�33A��C�>�@&ff�r�\?Y��A'�C�9�                                    Bxz�X
  �          @��@   �Q�@33A�p�C��=@   �l��?��
A��C�                                    Bxz�f�  	�          @��?�ff�^�R@)��Bp�C���?�ff�~{?�=qA�=qC�1�                                    Bxz�uV              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxz���  o          @��R@ ���c33@A���C��
@ ���~�R?�G�A��C�^�                                    Bxz���  �          @�\)@
=�`  @Q�A��
C��\@
=�{�?ǮA��C�q                                    Bxz��H  �          @���@G��R�\@%�Bp�C�O\@G��q�?�ffA��C���                                    Bxz���  �          @�  @{�i��@�
A�{C���@{��Q�?��HAl  C�xR                                    Bxz���  �          @�Q�@���qG�?��RA�G�C��f@�����?�{AW33C��R                                    Bxz��:  �          @�\)?�{����?��\AG33C��3?�{����<��
>uC�]q                                    Bxz���  �          @���?\����?��A�C�� ?\��\)>��
@xQ�C�aH                                    Bxz��  �          @�Q�?��R���H?��RAqp�C�}q?��R��Q�>u@<(�C�*=                                    Bxz��,  �          @�  ?�ff����?�G�Av=qC���?�ff��
=>�=q@QG�C��                                    Bxz��  �          @�ff?������?�(�A���C���?������R>��H@���C�:�                                    Bxz�x  �          @�p�?�(����\?O\)A!�C�aH?�(�����8Q��
=qC�=q                                    Bxz�%  �          @�?���
=?@  Az�C���?����þW
=�%C��q                                    Bxz�3�  �          @��?����G�?^�RA(��C�O\?����(�����
=C�q                                    Bxz�Bj  �          @�
=@z����?k�A5�C�W
@z���
=�����RC�\                                    Bxz�Q  �          @�{?������?��
AMG�C�Ф?�����G�=�\)?^�RC��H                                    Bxz�_�  �          @�ff?ٙ���G�?Y��A'�C�˅?ٙ����
�\)��C��)                                    Bxz�n\  �          @�
=?����\)?�(�Ap��C���?������>k�@4z�C�4{                                    Bxz�}  �          @���?�G���\)?�=qA��HC��{?�G���\)?z�@߮C�S3                                    Bxz���  T          @��?�33���\?fffA/
=C�s3?�33��p���G����C�AH                                    Bxz��N  �          @�  ?�(���?B�\Az�C�33?�(���\)����Mp�C�
                                    Bxz���  �          @���?��
��p�?��AQC���?��
����=L��?#�
C�J=                                    Bxz���  �          @��?�����  ?�z�A��HC�q?�����=q?c�
A&�\C�s3                                    Bxz��@  �          @�p�?�\)��Q�?�{A�p�C�aH?�\)��=q?Tz�A�
C���                                    Bxz���  �          @���@���  ?�\)A�(�C��@�����?(��@��RC�U�                                    Bxz��  �          @�\)?�z����@�
A�\)C�?�z����?�G�A;�
C�ff                                    Bxz��2  �          @�
=?�ff���\?�A�{C�ٚ?�ff��z�?J=qA�\C�>�                                    Bxz� �  �          @�(�?�G���=q?���A{\)C�9�?�G���  >k�@)��C���                                    Bxz�~  �          @��@33�u@Q�AиRC��@33��\)?�Q�Ab�\C���                                    Bxz�$  �          @�?�33���?�G�A�
=C���?�33��33?E�A�\C�˅                                    Bxz�,�  �          @�=q>L������������C�ff>L���_\)�R�\�-{C���                                    Bxz�;p  �          @��?�\��=q�
=q��33C�W
?�\�l���G
=�{C��                                    Bxz�J  �          @�z�>�����p��33��ffC��R>����tz��A����C�                                      Bxz�X�  �          @�p�>���녿��
��  C��>���Q��333�
��C�j=                                    Bxz�gb  �          @��?=p����\��=q���\C���?=p������7
=�C�/\                                    Bxz�v  �          @�  ?=p����R���R��
=C�|)?=p���\)�#�
��Q�C��)                                    Bxz���  �          @�  ?����������RffC���?������H�{���C�"�                                    Bxz��T  �          @�\)?�����ff�����_�C���?�����G������C�,�                                    Bxz���  �          @��?����Q쿂�\�<��C�.?����z������C���                                    Bxz���  �          @��?��R��Q�L����C�K�?��R��{��z����
C��                                    Bxz��F  �          @���?�G����ý��Ϳ��C���?�G����
��p��c
=C�f                                    Bxz���  �          @���?У���Q�����C�s3?У����H��  �e��C�                                    Bxz�ܒ  �          @�G�?������ý�\)�Tz�C�=q?������
���H�^�\C��f                                    Bxz��8  �          @�G�?����Q�u�(��C���?�������Q��[�C��3                                    Bxz���  �          @�G�?����Q�>�?��C�� ?�����Ϳ�G��9�C��{                                    Bxz��  �          @�G�?�z���Q�L�Ϳ(�C���?�z���������[�C��H                                    Bxz�*  �          @�Q�?�  ��Q�aG�� ��C���?�  ��=q��{�|z�C�
                                    Bxz�%�  �          @���?Ǯ����>�\)@P  C��?Ǯ��ff�^�R� z�C�(�                                    Bxz�4v  �          @��?�ff��G�>��H@��C���?�ff���׿.{��
=C��)                                    Bxz�C  �          @�=q?\����?�@�Q�C���?\��G��&ff���
C��3                                    Bxz�Q�  �          @�G�?�����Q�?#�
@��C��?������ÿ���33C��                                    Bxz�`h  �          @���?��H��(�?�  AiC�Q�?��H����=��
?s33C�{                                    Bxz�o  �          @��?�ff��z�@Q�A�ffC���?�ff����?�  A8Q�C�1�                                    Bxz�}�  T          @�(�?�Q�����?c�
A!�C�\)?�Q����
���R�`��C�>�                                    Bxz��Z  �          @�33?��H��G�>���@XQ�C�Ф?��H��
=�aG���
C��{                                    Bxz��   �          @���?�{��{?
=q@�Q�C�u�?�{���!G���ffC�z�                                    Bxz���  �          @��=#�
��(�@e�B�z�C��
=#�
��
@J�HBc�C�~�                                    Bxz��L  �          @�z�fff?�=q@�p�B��3C�f�fff�L��@���B��=C6�                                    Bxz���  �          @�(���>�@�\)B�p�C
�\���(��@�ffB��\Ch\                                    Bxz�՘  �          @�G�?녿Q�@���B�W
C�U�?녿��@���B��C�c�                                    Bxz��>  �          @���    ����@��\B��C��    �
=@w�Bj��C�f                                    Bxz���  �          @��H��{��  @��B�  C{!H��{�z�@���By��C�L�                                    Bxz��  �          @�녿���>�\)@��HB�p�C(33���ÿQ�@���B�CS��                                    Bxz�0  �          @�ff�}p�<�@���B���C1�q�}p����@�p�B��Cbff                                    Bxz��  �          @�G��#�
�+�@�p�B�aHCb5ÿ#�
��G�@��
B�#�Cy��                                    Bxz�-|  �          @�녾�׿�33@�ffB��\C~@ ����*=q@i��BV33C�f                                    Bxz�<"  �          @��?�33�9��@#�
B�RC���?�33�\��?�  A�33C�s3                                    Bxz�J�  �          @�p�@(��.�R@+�B��C�\)@(��Tz�?�z�A���C��3                                    Bxz�Yn  �          @�(�?���{@H��B?(�C�� ?���<(�@�RB  C��3                                    Bxz�h  �          @�ff?��\�
=@\��BUffC���?��\�J=q@/\)B33C��                                    Bxz�v�  T          @�{?�p��;�@0��B{C�G�?�p��a�?�A�{C�
=                                    Bxz��`  �          @�(�>����W�@HQ�B+�C��>������@
�HA�z�C�T{                                    Bxz��  �          @�{?+��j=q@�HB{C�4{?+�����?�33A�\)C��R                                    Bxz���  �          @�
=?��
�b�\@(�A�G�C���?��
�~�R?��HA~ffC���                                    Bxz��R  �          @��H@Q��S�
@
=AڸRC��@Q��o\)?�Q�Ap  C�AH                                    Bxz���  �          @��\@2�\�C33?���A�=qC�:�@2�\�\��?���A_
=C�|)                                    Bxz�Ξ  �          @��@>�R�9��?���A��C��)@>�R�Q�?��AR=qC�)                                    Bxz��D  �          @��@���N�R@
�HA�p�C��\@���l(�?�G�A~�HC�˅                                    Bxz���  �          @�ff@�`  @z�A�\)C��)@�z=q?��AW
=C�h�                                    Bxz���  �          @�\)@%��P  @p�A���C�7
@%��mp�?��A�C�c�                                    Bxz�	6  �          @�\)@'��[�?���A�
=C���@'��r�\?\(�A*=qC�T{                                    Bxz��  �          @�  @/\)�\(�?��A�
=C�C�@/\)�r�\?Tz�A ��C��                                    Bxz�&�  �          @��@*=q�fff?\A�Q�C�9�@*=q�w�?�@��C�<)                                    Bxz�5(  T          @�ff@"�\�r�\?��A`��C�� @"�\�}p�=�G�?�ffC�O\                                    Bxz�C�  �          @�ff@(��|(�?��
A~�RC���@(���z�>L��@   C��                                    Bxz�Rt  �          @��?����=q?\(�A1G�C���?�����;u�G�C���                                    Bxz�a  �          @x��?�
=�c33���
=C���?�
=�QG���p�����C��=                                    Bxz�o�  �          @w
=?�z��S33��{��C�aH?�z��8�ÿ�Q���  C��R                                    Bxz�~f  �          @y��?��H�L(�������C��f?��H�2�\��33����C���                                    Bxz��  �          @y��?����U������{C�� ?����7������C�^�                                    Bxz���  �          @u?���Q녿�=q���C��3?���/\)�����C���                                    Bxz��X  �          @��@�\�?\)?E�A8��C���@�\�E�u�Y��C�H�                                    Bxz���  �          @�\)?�����(�?�33Ab{C���?������ýu�@  C�G�                                    Bxz�Ǥ  �          @�=q?�  ����?�z�A`  C�` ?�  ������\)�G�C��                                    Bxz��J  �          @�G�?�(����\?�{AV=qC��{?�(����R���Ϳ�C��                                    Bxz���  �          @��?����(�?��A[�C�C�?�����׽��
�z�HC���                                    Bxz��  T          @�(�?ٙ���z�?�G�Ap  C���?ٙ���=q<�>��
C�0�                                    Bxz�<  �          @�=q?����?�z�A_
=C�k�?����R�L�Ϳ(�C��                                    Bxz��  �          @�Q�?�����Q�?��AL��C���?�����(��\)�ٙ�C�l�                                    Bxz��  T          @��
?޸R��z�?���AV�RC�b�?޸R���ý��Ϳ���C�\                                    Bxz�..  �          @���?�
=��(�?��
A�ffC�R?�
=��=q=�G�?���C���                                    Bxz�<�  �          @��?�ff����?�  A�Q�C��?�ff����>�{@�ffC�^�                                    Bxz�Kz  �          @�G�?�
=����?�Aq�C�U�?�
=��{<�>���C��                                    Bxz�Z   �          @���?��R��{?c�
A5�C���?��R���׾����o\)C���                                    Bxz�h�  �          @��?У����H?W
=A,  C���?У�������
����C��\                                    Bxz�wl  �          @�  ?������
?O\)A&ffC��)?�������Q����\C�w
                                    Bxz��  �          @��?�G��|��?�z�Ap��C��)?�G����
<��
>��RC���                                    Bxz���  T          @�Q�?˅��G�?�
=As33C��)?˅���R<#�
>#�
C�W
                                    Bxz��^  �          @���?ٙ���G�?�\)AeG�C�k�?ٙ���{�L�Ϳ��C��                                    Bxz��  �          @��@��qG�?�A�G�C���@���Q�>��R@z�HC���                                    Bxz���  �          @��@(��qG�?�ffA�G�C�@(��~�R>L��@ ��C�`                                     Bxz��P  �          @�\)@�b�\?��HA��
C���@�s33>��@��C�Ǯ                                    Bxz���  �          @��@\)�Z�H?�33A��C��q@\)�o\)?�RA=qC�p�                                    Bxz��  �          @�33@�R�n�R?�  AH��C�Ǯ@�R�w
=��Q쿔z�C�T{                                    Bxz��B  �          @�{@{�g�?&ffA�\C�q@{�j=q������{C��)                                    Bxz�	�  �          @�@\)�e?
=q@�  C�ff@\)�e�   �У�C�aH                                    Bxz��  �          @qG�?&ff�J�H���R�ŮC�Ф?&ff�'����#{C���                                    Bxz�'4  �          @`��>��H�ÿ�(���  C�8R>��&ff�33�%�C�y�                                    Bxz�5�  �          @g�>\)�X�ÿ�33��\)C�8R>\)�:�H�z��C�h�                                    Bxz�D�  T          @n{��Q��\(��������RC��)��Q��=p��Q��G�C�~�                                    Bxz�S&  �          @S�
?!G��A녿�����C��R?!G��%������p�C��{                                    Bxz�a�  �          @p  ?5�b�\�8Q��5G�C��\?5�L(����H���C�L�                                    Bxz�pr  �          @j=q?&ff�Y���\(��`Q�C�b�?&ff�@�׿�ff���C��                                    Bxz�  �          @tz�#�
�L(����H���HC�P��#�
�\)�333�=C��                                    Bxz���  �          @k���\�Mp����H��ffC��f��\�%��$z��1Q�C�t{                                    Bxz��d  �          @l�;����Z�H��\)��Q�C�uþ����8Q���
�(�C���                                    Bxz��
  �          @r�\��z��Z=q��=q����C��;�z��333�   �&��C�f                                    Bxz���  �          @w��
=q�\�ͿУ���C����
=q�5��#�
�&�C��3                                    Bxz��V  �          @w��.{�>�R��
�\)C��f�.{�
�H�E��U�
C|�                                    Bxz���  �          @z�H�333�B�\��\�C���333�\)�E��Rp�C|�                                     Bxz��  �          @w
=�}p��N{��{����C|޸�}p��&ff�\)�&��Cy&f                                    Bxz��H  �          @��׿��
�K��(��{C|����
����A��E�Cv��                                    Bxz��  T          @�  �Ǯ�:=q�  �	33Cq�ͿǮ���@���CQ�Ci�H                                    Bxz��  �          @~�R��=q�@���p��p�Cv#׿�=q�{�@  �D�\Co�                                    Bxz� :  �          @w���{�C33�����Cy�R��{�33�8Q��B�Ct0�                                    Bxz�.�  �          @j�H��\)�(Q��(��ffCnff��\)���*=q�:�\Ce�q                                    Bxz�=�  �          @l(���
=�z���R���Cj  ��
=��ff�4z��K��C^�)                                    Bxz�L,  �          @i�����R�I����G���33C�/\���R��R�'��9p�C�o\                                    Bxz�Z�  �          @l�Ϳ���,(������Cw���������>�R�W33Co��                                    Bxz�ix  �          @i���}p��/\)�(����Cz#׿}p����H�:=q�S�RCs8R                                    Bxz�x  �          @h�ÿ��
�   ��
��\Crٚ���
��Q��<���X��Ch�H                                    Bxz���  �          @^{?O\)�8�ÿ�������C��{?O\)�G�����3(�C���                                    Bxz��j  �          @j�H?����G���  ��33C���?����!G����!�C���                                    Bxz��  �          @���@   �L�ͿУ�����C�  @   �#�
�!G��33C�H                                    Bxz���  �          @��@$z��Dz�}p��_
=C���@$z��(Q�����Q�C�.                                    Bxz��\  �          @��@g���ff�����C���@g���
=�+����C��
                                    Bxz��  �          @�ff@XQ���R=�Q�?��RC��q@XQ���ÿ+��ffC�]q                                    Bxz�ި  �          @��
@O\)�!녾��R��  C��@O\)��
����n�RC�9�                                    Bxz��N  �          @���@K��'�������C�E@K�����Q���  C���                                    Bxz���  �          @�G�@=p��=p��L���-�C�� @=p��$z����(�C��                                     Bxz�
�  �          @�Q�@HQ��(�ÿ���d��C��\@HQ��(���ff���C��                                     Bxz�@  �          @���@C33�&ff����h��C�@C33�
=q�������C�\)                                    Bxz�'�  �          @��@P�׿��R���H���RC�S3@P�׿�Q��   ���
C��                                    Bxz�6�  �          @��
@P����ÿ����=qC�aH@P�׿У׿�33��z�C���                                    Bxz�E2  �          @�z�@L(�����(����C���@L(��������(�C��                                    Bxz�S�  �          @�z�@S�
�ff�����33C���@S�
��33��p��ȸRC��=                                    Bxz�b~  �          @�z�@Z=q�z�h���MC�XR@Z=q��Q��G���33C��{                                    Bxz�q$  �          @��
@Fff��
��������C��\@Fff��G������HC�33                                    Bxz��  T          @���@S�
�z�p���QG�C�y�@S�
��z��{��p�C��q                                    Bxz��p  �          @���@n{���
��G����
C�:�@n{��ff�xQ��YC���                                    Bxz��  �          @���@mp�����\)�y��C�H@mp��У׿Tz��:{C�%                                    Bxz���  �          @�ff@[���H=�Q�?��C�e@[����+��
=C��                                    Bxz��b  �          @��@e����R?   @�p�C�|)@e���\������C�&f                                    Bxz��  �          @��@*=q�@��B33C�� @*=q�+�?�
=A�C�]q                                    Bxz�׮  �          @s�
?�{�A녿�ff���C���?�{�"�\������RC�*=                                    Bxz��T  �          @��R�(��^�R��Q��G�C�׿(��QG���Q�����C��{                                    Bxz���  �          @����(Q��\)����=qCS���(Q�G��2�\�5��CDs3                                    Bxz��  �          @���q녿=p��
=���
C?)�q�=L���{�C3G�                                    Bxz�F  �          @�������aG����ƸRC?�
�����B�\�p���33C6�                                     Bxz� �  �          @��H���ÿn{�33����C@E���þk��\)���HC7                                    Bxz�/�  T          @��������p����хCH\�����8Q���R����C>�                                    Bxz�>8  T          @������H����33��
=CF���H�(������C=!H                                    Bxz�L�  �          @�G��\)�޸R���R��ffCK�H�\)��G��(���CB:�                                    Bxz�[�  l          @����|(��녿��
���HCOJ=�|(��������\CF�                                    Bxz�j*  �          @�  �qG���Ϳ\���
CU�qG�������R���CM�{                                    Bxz�x�  �          @�  �j=q�'
=���H��ffCW���j=q���R��R��p�CP��                                    Bxz��v  �          @�\)�HQ��J=q�����33CaE�HQ��\)�\)��
=CZu�                                    Bxz��  �          @�p��:�H�P  ��=q���RCc�q�:�H�#�
�#33�
=C]+�                                    Bxz���  �          @��R�Dz��P  ��
=���
Cb���Dz��'
=�=q��\)C\\)                                    Bxz��h  �          @�p��2�\�_\)��������CgY��2�\�7
=��H��Ca��                                    Bxz��  �          @��R�5�c�
��Q��j�RCgff�5�>{��\���CbJ=                                    Bxz�д  �          @���.{�h�ÿ�ff�~�RCiE�.{�@����H��ffCc�R                                    Bxz��Z  l          @�
=�Fff�Z�H�^�R�*{Cc�)�Fff�<�Ϳ�Q��Ù�C_�{                                    Bxz��   �          @��   �vff�fff�2ffCl��   �U������
Ci{                                    Bxz���  T          @�z��*�H�n{����CjO\�*�H�W
=��z���
=Cg��                                    Bxz�L  �          @�=q�4z��^�R�p���<��Cf��4z��>{�����Cbu�                                    Bxz��  l          @���W��E��fff�2{C^xR�W��'
=��\)����CY�                                    Bxz�(�  �          @�=q�6ff�S�
���\�Qp�Ce=q�6ff�1���
�ڏ\C`J=                                    Bxz�7>  �          @����Z=q�&ff��33��ffCYW
�Z=q���R�(���CR:�                                    Bxz�E�  �          @�(��e�#�
��\)���CW� �e���H�����p�CP�
                                    Bxz�T�  �          @�p��i���!G���\)���CV���i���������\)CO�q                                    Bxz�c0  �          @�p��l������ff��p�CT�H�l�Ϳ��H�����CL�{                                    Bxz�q�  �          @��H�3�
�[������^=qCf���3�
�7
=�����Ca}q                                    Bxz��|  �          @�  <#�
���z���C�<#�
�~{�����HC��                                    Bxz��"  �          @�Q�8Q����H�+��
=C�G��8Q��w
=�ff����C��3                                    Bxz���  �          @�ff?k���G���z��qG�C��?k��|(���(�����C���                                    Bxz��n  �          @�
=�=p���Q쿁G��Qp�C�R�=p��j�H����G�C�Q�                                    Bxz��  �          @�Q�?O\)���׾�33���
C���?O\)�j�H�ٙ���ffC�@                                     Bxz�ɺ  �          @���?�����녾�Q���(�C��3?����l(���(����C��=                                    Bxz��`  �          @���?�ff�������z�C��?�ff�|(��������\C���                                    Bxz��  �          @��\?��R��\)������C���?��R�w
=��  ����C��=                                    Bxz���  �          @��\?�\��p�>�  @Q�C�L�?�\������\����C�p�                                    Bxz�R  �          @�33?�=q��33�:�H�z�C���?�=q�u����C��                                    Bxz��  �          @���>\��(�������C��f>\�P���Fff�-�\C�Z�                                    Bxz�!�  �          @�  >�33�����33���\C�g�>�33�R�\�>{�'�\C�                                      Bxz�0D  �          @���>�\)�~�R�
=q��33C�>�\)�?\)�Y���B(�C���                                    Bxz�>�  �          @��������������ffC������J�H�G
=�0p�C���                                    Bxz�M�  �          @�  ��  �H�ÿ.{�,  Cp�\��  �-p���p���Cm�                                    Bxz�\6  �          @�=q�G��R�\?   @��HCb���G��P�׿+��33CbE                                    Bxz�j�  �          @��\�333�c33?��@�33Cg�3�333�b�\�+��
=Cg�)                                    Bxz�y�  �          @���%�vff?Tz�A"�\Cl{�%�y���\)���
Clh�                                    Bxz��(  �          @�p��G���Q�?333A	p�Cp���G���  �@  �(�Cpu�                                    Bxz���  �          @��Ϳ�����33>#�
@�Cy�����x�ÿ�����\Cx                                    Bxz��t  �          @�{�����\)>k�@<��C}
�����G����\��p�C|W
                                    Bxz��  �          @�\)�p������?�Q�Az�HC��f�p����녾�{��C�޸                                    Bxz���  �          @�\)�����������
��  C�  �����\)�˅���C~�q                                    Bxz��f  �          @��B�\�w
=������C�s3�B�\�G��*�H��C�.                                    Bxz��  �          @�zᾨ���5��O\)�B�RC�����ÿ�  ����L�C��\                                    Bxz��  �          @��
?�{����xQ���C�� ?�{>�Q�����p�Aj�H                                    Bxz��X  �          @�  ?��\�h���q�C��q?����|��(�C���                                    Bxz��  T          @��R?!G��33�\(��](�C�� ?!G��k�����
=C��                                    Bxz��  �          @�{�Q��hQ�������C����Q��"�\�a��S�C|\                                    Bxz�)J  �          @�Q�#�
�fff�(Q��C���#�
���o\)�`��C�                                    Bxz�7�  �          @�  �E��^�R�.{��\C���E��G��r�\�f�\C{O\                                    Bxz�F�  �          @������`  �6ff�33C��R���  �z=q�m�HC�ff                                    Bxz�U<  �          @�p����R�G��\���?
=C�&f���R�����HB�C��R                                    Bxz�c�  �          @��ÿ�\�dz��)���ffC�����\���p���d�\C��f                                    Bxz�r�  �          @������׿fff�2�RC����x���{� C���                                    Bxz��.  �          @�=q���u�
�H��z�C�&f���2�\�Z�H�I  C��)                                    Bxz���  �          @�33��Q������
=���C�ff��Q��HQ��R�\�9z�C�5�                                    Bxz��z  �          @���
=q���������C�Ff�
=q�N{�L���1�C�4{                                    Bxz��   �          @�ff��{��
=��(���Q�C|.��{�Y���9����\Cx=q                                    Bxz���  �          @����Q���zΌ�����
C�����Q��c�
�<���
=C�)                                    Bxz��l  T          @�zῘQ���녿����\Q�C~�{��Q��g
=�%�	(�C{�=                                    Bxz��  �          @��׿�G���녿��\�D��Cw�Ϳ�G��h���!G�����Ct5�                                    Bxz��  �          @�33�C33�I����
=����Ca��C33�=q�\)�p�CZE                                    Bxz��^  �          @��\�}p��  ���
����CQ�
�}p����
�  ��\)CI!H                                    Bxz�  �          @��\�k��{����
=CU��k������,(���\CKxR                                    Bxz��  �          @��\�P  �P�׿����Ca)�P  �   �!G����HCY��                                    Bxz�"P  �          @��H�b�\�Dzῗ
=�b=qC\��b�\��H��R���
CVT{                                    Bxz�0�  �          @����c33�Y�������  C_���c33�@�׿����
=C\=q                                    Bxz�?�  �          @�Q��ff�^{�J�H�(��C�P���ff�33��\)�~��C���                                    Bxz�NB  �          @��׿�G��`���<(��Cz5ÿ�G��
�H�����h=qCo��                                    Bxz�\�  �          @�Q�У��dz��+��	33Cuc׿У��z��s�
�Up�Cj��                                    Bxz�k�  �          @��
�����k��#33��ffCs�f�����{�n�R�JffCi�=                                    Bxz�z4  �          @�33������aG��,��Cv�����w
=�����  Ct�)                                    Bxz���  
�          @��R�z��~{�����pQ�Cr}q�z��N�R�&ff�ffCmff                                    Bxz���  �          @�=q��ff����E��{C�B���ff�����p���{C���                                    Bxz��&  �          @��>u���R������C���>u�Z�H�[��4  C��R                                    Bxz���  �          @��
��p���  ���
��  C����p��`  �U�-��C�f                                    Bxz��r  �          @��
�������
���
�y�C|�=�����dz��5��Q�Cy8R                                    Bxz��  �          @��Ϳ�{�����=q����C|����{�_\)�HQ��ffCx                                    Bxz��  �          @�p���G���������ffC�lͿ�G��K��g
=�=  C|n                                    Bxz��d  �          @��n{����H�뙚C��{�n{�<���tz��K�HC|u�                                    Bxz��
  �          @�ff�����=q�&ff��33C~�q����1��|���S�\Cx�{                                    Bxz��  �          @��R�W
=�j�H�K�� �HC��׿W
=�����H�v��Cx�
                                    Bxz�V  �          @�ff    ��(����R��ffC��R    �mp��H��� �RC��
                                    Bxz�)�  �          @��R>�\)���H��  �9�C��=>�\)���H�1G��  C��R                                    Bxz�8�  �          @�\)���
��\)���
��G�C�⏼��
�r�\�N{�!z�C��)                                    Bxz�GH  T          @�
=<#�
��p���
=��z�C�3<#�
�j�H�U�)=qC�R                                    Bxz�U�  �          @�G�>aG����Ϳ���p�C�aH>aG��c�
�c�
�3��C��                                    Bxz�d�  �          @��>�
=��=q������C�xR>�
=�w
=�QG�� z�C�{                                    Bxz�s:  �          @�����H�z�H�C33���C�ff���H�(���=q�p
=C�Ff                                    Bxz���  �          @������mp��J�H�!G�C�޸����p�����{{C���                                    Bxz���  �          @��R�0���b�\�W��,\)C���0�׿��H��\)33Cz�                                     Bxz��,  �          @�
=��z��b�\�]p��1�C��\��z�������C���                                    Bxz���  �          @��R�fff�Dz��qG��F�HC}�f�fff��\)��p�W
Cl��                                    Bxz��x  �          @��\���\�)�����
�Z33CtY����\�W
=���\��CUk�                                    Bxz��  �          @��u�%����\�`  Cy�=�u�J=q��Q��3C[aH                                    Bxz���  �          @��
�#�
�1��|(��XffC����#�
���
���R=qCn
                                    Bxz��j  �          @�=q��{�1��z=q�YG�C��;�{������3C|�                                    Bxz��  �          @��H=�\)�C�
�p  �K
=C��f=�\)������z�� C�y�                                    Bxz��  T          @�(��u�4z��z=q�X��C���u�����ff��C��)                                    Bxz�\  l          @�(�>���'�����e�RC��)>���L������ ǮC�9�                                    Bxz�#  �          @��
>��������y33C���>��\��33ªk�C�K�                                    Bxz�1�  �          @�(�>u����Q��s
=C��=>u�   ��33¦��C��                                    Bxz�@N  �          @��H>#�
��
=��p���C�\)>#�
�����\°#�C��                                    Bxz�N�  �          @�z�?\)���H��(��RC�p�?\)>��H���\¢��B$��                                    Bxz�]�  �          @��>�
=��=q��ff��C���>�
=>��R���R§L�B�                                    Bxz�l@  �          @�
=>B�\���R��G���C���>B�\����
=®u�C��3                                    Bxz�z�  �          @��\?:�H�fff������C��{?:�H?��\��(���BY
=                                    Bxz���  �          @��þ���)���}p��`��C�/\����Y����{u�C}�                                    Bxz��2  �          @��
�u�[��@  �$\)C���u���H���
�HC���                                    Bxz���  �          @��H�#�
�G��QG��9G�C����#�
�Ǯ��  � C�3                                    Bxz��~  T          @��=����<(��^�R�G  C��)=��Ϳ������ C�4{                                    Bxz��$  T          @�Q�   �1��`  �L�C��3�   ��33��=q�qCv��                                    Bxz���  �          @��������  �|(��pQ�Cf\)�������
����HC4�H                                    Bxz��p  �          @�  �z������o=qCA��z�?����\)�dz�C�                                    Bxz��  �          @�Q��
=q�5�����u�CF+��
=q?��\���\�nCǮ                                    Bxz���  �          @������#�
��p��p33C7�=���?��
�x���W  Ch�                                    Bxz�b  �          @�ff�:=q?p���xQ��O33C"#��:=q@�H�K��   C5�                                    Bxz�  �          @�z�� ��?
=q��(��h��C'�3� ��@
=q�c�
�<z�CQ�                                    Bxz�*�  �          @�(��z�?&ff��ff�p��C$xR�z�@�\�e��>�RCk�                                    Bxz�9T  �          @�z���R>�����H�z�C0&f��R?���y���T��C�                                    Bxz�G�  �          @�(��
=q�����33�}C:��
=q?��
���\�d�C��                                    Bxz�V�  �          @��H���R��\)��z��RC;녿��R?��
���
�j�\C��                                    Bxz�eF  �          @��\��\�G���G��|(�CI
=��\?��
����v��C.                                    Bxz�s�  T          @������z���{�n�HC:����?�Q��|(��Y�\CL�                                    Bxz���  �          @�=q��
�p�����H�j��CJ(���
?O\)���
�m(�C �f                                    Bxz��8  �          @�=q��ÿ�ff�u�U�CT����>#�
��z��o�C0J=                                    Bxz���  �          @�G��0�׿�ff�e��ACQY��0��=L���y���Z�C3\                                    Bxz���  �          @�Q��{�L(��&ff�z�Cs8R��{���k��aCd�{                                    Bxz��*  �          @�z���H�>�R�Dz��'G�Cp0����H���H�����sp�C\u�                                    Bxz���  �          @�33��Q��o\)�Q���  C|LͿ�Q����mp��Wz�Cs�
                                    Bxz��v  �          @�=q�����}p���G���  C~\�����E�1�� �Cy�f                                    Bxz��  �          @��H>����R>���@w
=C��)>���{�Ǯ���
C��=                                    Bxz���  �          @��H?.{���R?
=@�RC�S3?.{��녿�ff��33C�y�                                    Bxz�h  �          @��
?.{��G��u�E�C�C�?.{�����
=��G�C��{                                    Bxz�  �          @��H?L������\����C��?L���{��p�����C�Ǯ                                    Bxz�#�  �          @�z�?:�H��p���\)�`��C���?:�H�c�
�6ff��C��f                                    Bxz�2Z  T          @�p�?p����G��333�(�C�33?p���g
=�=q��\C�P�                                    Bxz�A   T          @���?u�H��@`  B;33C��=?u���?�\)A�  C�N                                    Bxz�O�  �          @��H?B�\�W�@XQ�B1�C�Z�?B�\��
=?�A��C���                                    Bxz�^L  �          @�=q?=p��Tz�@Y��B3�C�J=?=p���{?ٙ�A�  C���                                    Bxz�l�  �          @��?E��X��@S�
B.(�C�j=?E���
=?˅A�z�C���                                    Bxz�{�  �          @��H?(���mp�@B�\B��C��?(����z�?��HAg�C��                                    Bxz��>  �          @���?Q��\)@\)A�\)C��R?Q����?
=@��HC�                                    Bxz���  �          @��?�\��?�Q�A�C�xR?�\���<#�
=�\)C�&f                                    Bxz���  �          @��R?Q��s�
@&ffB�
C�
=?Q�����?B�\A�C�{                                    Bxz��0  �          @�Q�?����tz�@!G�B �RC��\?�������?.{A  C�ff                                    Bxz���  �          @�\)?�ff��33@ffAӮC�/\?�ff��33>8Q�@{C�l�                                    Bxz��|  �          @��?xQ��~{@\)A�C��3?xQ�����>�33@��\C���                                    Bxz��"  T          @��?:�H���R?�z�ArffC��?:�H��녿:�H�  C�Ф                                    Bxz���  �          @��?�����  ��=q�QG�C�y�?����~{�	������C�l�                                    Bxz��n  �          @���?�����Q�?��
A�=qC���?������Ϳ#�
��\)C�XR                                    Bxz�  �          @��H?����J�H@EB =qC�)?�����{?���A���C��                                    Bxz��  �          @�G�?��H���
?�Q�A���C��f?��H����<#�
=�Q�C��f                                    Bxz�+`  �          @��H?�����G�@�A���C�s3?������
>�33@��C�S3                                    Bxz�:  �          @��\?k���
=@
�HA�p�C�%?k����>.{?�(�C�|)                                    Bxz�H�  �          @���?xQ����@(�AڸRC���?xQ�����>aG�@+�C��                                    Bxz�WR  �          @���?��H����@33A�C��?��H��G�=�?�  C���                                    Bxz�e�  �          @���?�{���@	��A�  C��?�{��z�>8Q�@�C��{                                    Bxz�t�  �          @���?G����@
=A�C�ff?G���
=>\@��C���                                    Bxz��D  �          @��\?.{���
@�A�C��R?.{��Q�>�(�@��C�
                                    Bxz���  �          @��H?��\��33@ffA��C���?��\��
=>�Q�@�G�C��                                    Bxz���  �          @�=q?p������@�A��C��f?p����ff>�ff@�C��                                    Bxz��6  �          @���?k��|(�@!�A��C��3?k�����?
=@ᙚC���                                    Bxz���  �          @�=q?������@#�
B z�C�:�?�����?�@�=qC���                                    Bxz�̂  �          @��?5��33@��A�  C��?5���>���@�Q�C�J=                                    Bxz��(  �          @���?fff���\@A�p�C�:�?fff��{>�{@��HC�q�                                    Bxz���  �          @���?����s�
@!�A��\C�j=?�������?!G�@�=qC�޸                                    Bxz��t  �          @�=q?�z�����@�
A�RC�
=?�z���(�>�{@���C�
=                                    Bxz�  �          @���?xQ����@��A��
C��=?xQ����>L��@�C���                                    Bxz��  �          @���?u��=q@�\A�C���?u���>���@c33C�˅                                    Bxz�$f  �          @���?�{����@	��AָRC�N?�{��=q>B�\@�RC�Ff                                    Bxz�3  �          @�=q?��H�z�H@��A�C�:�?��H��=q>�G�@�Q�C��H                                    Bxz�A�  �          @��H?�(��{�@��A�Q�C�Ff?�(����\>�ff@��HC���                                    Bxz�PX  
�          @�G�?�=q���@	��AՅC��?�=q���H>#�
?�C�\                                    Bxz�^�  �          @���?�p��x��@�
A�Q�C�l�?�p���Q�>\@�=qC�R                                    Bxz�m�  �          @���?�G��s33@�
A��HC�t{?�G���p�>��@�
=C�޸                                    Bxz�|J  �          @�G�?˅�z=q@{A��C�\?˅���>�\)@Y��C��                                     Bxz���  �          @��?����o\)@�A���C�/\?�������>��H@��
C�l�                                    Bxz���  �          @�33@Q��%�@p�A�=qC�޸@Q��Y��?�=qAM��C��
                                    Bxz��<  �          @�33@B�\�I��@ ��A�33C��R@B�\�n{>�
=@��C���                                    Bxz���  T          @���?�Q��~�R@  A߮C��=?�Q����>�=q@K�C���                                    Bxz�ň  �          @���?�  �w
=@��A���C�,�?�  ���R>���@|(�C���                                    Bxz��.  �          @��
?����mp�@!G�A��C�\?�����{?!G�@�C�"�                                    Bxz���  �          @��
@$z��Tz�@��A�  C��@$z����?:�HA
�\C�.                                    Bxz��z  �          @��
@.�R�4z�@6ffB�
C��@.�R�r�\?��AzffC��                                    Bxz�    �          @�z�@{�(Q�@R�\B)ffC��)@{�s�
?��A���C�xR                                    Bxz��  T          @���@#33��@\��B7{C��{@#33�_\)@�AхC�
                                    Bxz�l  �          @�ff@z�У�@�G�B[�C�g�@z��P��@9��B�
C��{                                    Bxz�,  �          @�p�@����@\)BXC���@��P  @6ffB\)C�                                    Bxz�:�  �          @�p�@  �G�@vffBO��C�@  �aG�@#33A��C�J=                                    Bxz�I^  �          @�p�?�ff�(�@x��BUz�C�:�?�ff�z�H@��A��C��\                                    Bxz�X  �          @���?�\)��
=@��
Bq�C�Q�?�\)�G�@C�
B 33C�}q                                    Bxz�f�  �          @�ff?�z�.{@�B�\C�.?�z��,(�@x��BO�C�Ф                                    Bxz�uP  �          @�p�?��Ϳ���@�=qBy33C�xR?����W
=@I��B C���                                    Bxz���  �          @�z�?�33��\@xQ�BR33C���?�33�q�@��A��\C�W
                                    Bxz���  �          @�p�?��
��@|(�BY�C��\?��
�n{@"�\A��C�Ǯ                                    Bxz��B  �          @�p�?�{�
=@vffBO�HC��?�{�u@Q�A�\)C��3                                    Bxz���  �          @��
?�G��n�R@�A�G�C��?�G�����>�ff@���C��=                                    Bxz���  �          @��\?�=q��33?��
A�\)C�f?�=q���R��  �:�HC�%                                    Bxz��4  T          @�33?�p��qG�@�RA�33C�Q�?�p���\)?�@ƸRC���                                    Bxz���  �          @�33?�\�n{@#33A�(�C���?�\��
=?(�@�p�C���                                    Bxz��  �          @�33?�
=�o\)@%B ��C��?�
=��  ?�R@���C�8R                                    Bxz��&  �          @�=q?�\)�c33@4z�BG�C�L�?�\)��{?h��A/33C�                                    Bxz��  �          @��H?���aG�@2�\B�
C�z�?������?fffA+\)C�3                                    Bxz�r  �          @�=q?У��e�@1�B�
C�5�?У���ff?\(�A$Q�C�
=                                    Bxz�%  �          @��H?�p��|(�@(�Aי�C��R?�p���  >.{@ ��C��                                     Bxz�3�  �          @��\?�����Q�?�
=A�G�C�e?�����ff��Q쿈��C�K�                                    Bxz�Bd  �          @��\?�p��w
=@�A�{C��
?�p�����>\)?��C�                                      Bxz�Q
  �          @�=q@{�n{@A�ffC�c�@{��Q�>8Q�@�C���                                    Bxz�_�  �          @��H@ ���i��@��A�p�C�l�@ �����?�@�
=C�^�                                    Bxz�nV  T          @�33?�p��}p�?��HA���C�G�?�p�����u�&ffC��                                    Bxz�|�  �          @�z�?�p����
?�p�A�C���?�p����R��{�|(�C���                                    Bxz���  T          @��@ ����Q�?�G�A��C�L�@ �����
����C33C�T{                                    Bxz��H  �          @�p�?�  �33@�=qB`{C���?�  �z=q@%�A��C��                                     Bxz���  �          @�=q?�G��@��@W�B3�HC�P�?�G���
=?�\)A�33C���                                    Bxz���  �          @���?�ff� ��@hQ�BFQ�C��q?�ff�x��@�A�G�C�s3                                    Bxz��:  �          @���?��%�@g
=BCffC�z�?��|(�@�A�=qC�K�                                    Bxz���  �          @���?޸R�333@^{B:
=C���?޸R��=q?�A�\)C���                                    Bxz��  T          @��?���[�@P��B(p�C��
?����G�?���A|��C��                                    Bxz��,  �          @��
?�{�aG�@E�B��C���?�{��G�?�\)AU�C�XR                                    Bxz� �  �          @�p�?��H�c�
@:�HB��C��R?��H��Q�?uA4��C�h�                                    Bxz�x  �          @�p�?\�l(�@8Q�Bp�C�33?\��33?^�RA"ffC�%                                    Bxz�  �          @�
=?��R�e�@N{B!�\C���?��R���?�(�Ab�\C�q�                                    Bxz�,�  �          @��?��H�^�R@N{B$ffC��R?��H��=q?�  Am��C�l�                                    Bxz�;j  �          @���?����@  @W�B:(�C��?�����
=?�{A�ffC�S3                                    Bxz�J  T          @���?B�\�XQ�@H��B(��C�O\?B�\��ff?�p�As�
C��{                                    Bxz�X�  �          @��\?@  ��  @   A�=qC�W
?@  ��ff>�p�@�{C��\                                    Bxz�g\  �          @���>�����?�  AG�C���>���
=�����n=qC��{                                    Bxz�v  �          @���?���{@�Aڏ\C��R?���
=    ���
C�0�                                    Bxz���  �          @�=q>������@�\A�z�C�N>������>#�
?�z�C��H                                    Bxz��N  �          @�=q�L����
=?�  A��\C���L�����׿��ǮC��f                                    Bxz���  �          @����\)��=q@z�A�(�C�{�\)��G��.{�z�C�+�                                    Bxz���  �          @��ü#�
���
?�{A��C���#�
��\)�\����C��\                                    Bxz��@  �          @����L����\)?��A�Q�C��þL����\)�!G����HC��f                                    Bxz���  �          @��H>�G�����>��@��
C���>�G���ff��ff���C�ٚ                                    Bxz�܌  �          @�33>B�\��Q�?Tz�Ap�C�  >B�\��33��p�����C�*=                                    Bxz��2  �          @��>����
=?!G�@��C���>����
=�����Q�C��q                                    Bxz���  �          @�G�?���
=?�@�(�C�33?���p���p�����C�h�                                    Bxz�~  T          @��׾#�
��
=>�@�33C���#�
���Ϳ�\��\)C���                                    Bxz�$  �          @���?E���>�ff@��C��?E���33������\C��                                    Bxz�%�  �          @���?L����ff?�@�C��?L�����Ϳ޸R��  C�                                      Bxz�4p  �          @���?c�
����?#�
@�p�C�j=?c�
�����{��p�C��{                                    Bxz�C  �          @���?����33?\)@أ�C���?�����\��z����RC�                                    Bxz�Q�  �          @���?u���>�
=@��\C��
?u��=q����C�J=                                    Bxz�`b  �          @�G�?���(�>�33@�  C��?���  ������ffC���                                    Bxz�o  �          @���?�ff��p�>aG�@+�C�U�?�ff��\)���R��ffC��R                                    Bxz�}�  �          @���?���(�=�Q�?�\)C��?���(���υC��H                                    Bxz��T  �          @���?����(�>�?ǮC��H?��������
���C�]q                                    Bxz���  �          @���?z�H��p�>aG�@*�HC��?z�H��\)�   ��C���                                    Bxz���  �          @���>�Q���ff?��@ָRC�/\>�Q������(���  C�T{                                    Bxz��F  �          @��
>�ff���?!G�@�\C���>�ff��\)������C�ٚ                                    Bxz���  �          @�z�>k�����?:�HA	��C�ff>k����\��\)���C�w
                                    Bxz�Ւ  �          @�z����r�\�(�� ��Cq8R���C33�z�� �Ck��                                    Bxz��8  �          @�����33�����G�����CDW
��33��G��ff���C5xR                                    Bxz���  �          @��
��(���G��
=�ͅCH���(������$z�����C8�{                                    Bxz��  �          @�����(�����
��Q�CJ���(��   �&ff� {C:�f                                    Bxz�*  �          @�{������ÿ�z���ffCK:�����Q�����Q�C>�                                    Bxz��  �          @�z���Q�����]�CN+���Q쿥��G��ÅCDٚ                                    Bxz�-v  �          @�������{�޸R����CH�����
=��\���C;�q                                    Bxz�<  �          @�����\)���Ϳ������RCE����\)���
�  ���C8\)                                    Bxz�J�  �          @����Q쿪=q��G���z�CEG���Q쾨������=qC8s3                                    Bxz�Yh  T          @����(���\������p�CNW
��(��������(�CC�                                     Bxz�h  �          @��\���ͿY����\��  C>�f����=�Q���H���C2��                                    Bxz�v�  �          @��H��33��
=�Ǯ���\CF@ ��33���33��
=C:�{                                    Bxz��Z  �          @����Q��G��������CN����Q�fff�%�� C@�H                                    Bxz��   �          @�33�z=q�  �ٙ����CQ���z=q���#�
� �CD�f                                    Bxz���  �          @�Q����ÿ�
=��p����CF�����ÿ\)���R���C;c�                                    Bxz��L  �          @����l���p���=q��  CR�)�l�Ϳ����*=q�
�\CD.                                    Bxz���  �          @�{���
��ÿL�����CN)���
��  ��(���CF��                                    Bxz�Θ  �          @�������\�����Q�COaH�����ÿ��H���CJQ�                                    Bxz��>  �          @���*�H�P  ��\����Cf�\�*�H�'
=�   ����C`\)                                    Bxz���  T          @���?@  �s�
��  �f�HC��\?@  �7��,���){C�O\                                    Bxz���  �          @�z��p��|�Ϳ�p��mCn��p��8Q��>{�p�Ce�                                    Bxz�	0  �          @�Q���{�'
=��G����HCS����{���У����CNk�                                    Bxz��  �          @����{���L�Ϳ
=qCIh���{�˅�k��(z�CF                                    Bxz�&|  �          @�Q���  ��  >�Q�@�CE����  ���R������=qCEp�                                    Bxz�5"  �          @�
=�������\��Q쿅�CBǮ�������ÿ.{��{C@��                                    Bxz�C�  �          @�����׿p�׿
=q��G�C?+����׿녿k��-C:�\                                    Bxz�Rn  �          @����׿��;�G���p�CA����׿B�\�k��,(�C=�                                    Bxz�a  �          @�
=���׿�������CC&f���׿aG�����J=qC>k�                                    Bxz�o�  �          @��R���ÿ�  <��
>aG�CB�
���ÿ��Ϳz���  C@�R                                    Bxz�~`  �          @�ff��\)�����\)�QG�CCs3��\)��G��^�R�#�C@{                                    Bxz��  �          @�Q�������R�z�����CEu�������
��p��b�RC@@                                     Bxz���  �          @�Q������녿O\)�p�CDL�����Tz῰����
C=�H                                    Bxz��R  �          @��R��
=���ÿ�  �:�RC@����
=���������C9�{                                    Bxz���  �          @������{���H�a��CAW
���Ǯ�˅���C8Ǯ                                    Bxz�Ǟ  �          @������\��녿O\)�Q�C8�����\<��h���*ffC3�
                                    Bxz��D  T          @�=q��\)�Tz�������C=����\)�
=q�B�\�	�C:&f                                    Bxz���  �          @����{�fff��\����C>O\��{�
=q�aG�� ��C:B�                                    Bxz��  �          @������\��{����\)CC�q���\�u��=q�E��C?@                                     Bxz�6  �          @�33���׿У׿:�H�(�CF�����׿��ÿ���  C@�f                                    Bxz��  �          @�z���{��ff�z��θRCB����{�\(������K�
C=ٚ                                    Bxz��  �          @�ff�����p��\��ffCAٚ����fff�n{�%p�C>+�                                    Bxz�.(  �          @������Ǯ�aG���CE������p�׿��
��{C>��                                    Bxz�<�  ;          @�������޸R��ff��Q�CI:�����&ff��H����C<L�                                    Bxz�Kt  T          @����z��   �	���ȸRCM� ��z�5�6ff���C=�3                                    Bxz�Z  
Z          @��H�~{��(��\)���CNaH�~{���HQ���C;p�                                    Bxz�h�  �          @�����\���
���ݙ�CD�����\<��,����33C3��                                    Bxz�wf  T          @��\�w��'
=� ����{CU�q�w�����AG���
CF�R                                    Bxz��  
�          @��\�?\)�hQ��\����Cfz��?\)�z��S33�$p�CY�                                     Bxz���  �          @����n�R�#�
����G�CVu��n�R��p��C�
��\CFE                                    Bxz��X  �          @�G��n�R�33�p���ffCS�3�n�R�Q��P���"=qC@k�                                    Bxz���  "          @����8���N�R�{��p�Cd=q�8�ÿУ��n�R�Ap�CQn                                    Bxz���  T          @����)���[��!����
ChJ=�)����\�xQ��J�CU                                    Bxz��J  
�          @��H�HQ��K��p���33Caff�HQ�ٙ��^�R�133CP��                                    Bxz���  T          @�33�U�C�
�z����C^��U��ff�aG��.��CL޸                                    Bxz��  �          @�(��)���r�\������Ck��)���33�i���8�C\�3                                    Bxz��<  T          @����I���`  ��\���Cd��I���z��^�R�*��CUT{                                    Bxz�	�  
�          @�p��S�
�Y����
���Ca���S�
��(��\���'ffCR��                                    Bxz��  
�          @�z��Z�H�Q녿������\C_�{�Z�H���S33� 33CQT{                                    Bxz�'.  �          @�=q�^{�@  ������C\�\�^{�˅�Tz��#�CL�=                                    Bxz�5�  �          @����C�
�:�H�#33����C_���C�
����i���>ffCK�                                    Bxz�Dz  T          @�G��4z����c�
�5��CXaH�4zὣ�
���
�^\)C5��                                    Bxz�S   �          @��H�N{�9���!���{C^\�N{��ff�g��8�HCJ
=                                    Bxz�a�  �          @���hQ��'
=�   ���CW�q�hQ쿇��\���)��CDJ=                                    Bxz�pl  �          @��R�h���  �<(��
{CS�q�h�ÿ��j�H�3�C<J=                                    Bxz�  �          @�p��hQ���
�L(����CN{�hQ�    �i���4�C3�3                                    Bxz���  
�          @��
�2�\��ff�y���J��CQ��2�\?������^�C(�=                                    Bxz��^  �          @�(��Tz��*=q�5���CZ���Tz�p���q��>�CC�                                     Bxz��  �          @��
�7
=�#�
�Vff�$=qC]�{�7
=�����p��[�
C?޸                                    Bxz���  �          @���-p��'
=�^�R�+  C_�f�-p��z�����e  C@                                    Bxz��P  T          @����A��\(���
=�e�Cd�
�A����-p��z�CZ��                                    Bxz���  T          @���B�\�Q�?&ffA  Cc+��B�\�K�����Tz�CbE                                    Bxz��  
�          @��R�a��U��(���=qC_xR�a��z��H���  CRs3                                    Bxz��B  T          @����[��`  ��G���=qCa�)�[��(��P  �z�CT��                                    Bxz��  T          @���+��\)���
����Cl+��+��/\)�R�\�"��Ca�H                                    Bxz��  �          @�{�;��w������z�Ch޸�;��\)�]p��'�
C\h�                                    Bxz� 4  ;          @�p������&ff��Cu����p  �6ff�Q�Cp�\                                    Bxz�.�  	          @�(���
=���R����ffCw�R��
=�xQ��-p�� �Cs�\                                    Bxz�=�  T          @��R����33��ff��Cy�Ϳ���Q��0��� ��Cu�R                                    Bxz�L&  
s          @�p��  ��
=���R�]�CsB��  �R�\�QG��\)Ck��                                    Bxz�Z�  ;          @���@  �{���z���ffCh�3�@  �'
=�W��!G�C])                                    Bxz�ir  "          @���l(��2�\�+���Q�CY)�l(���\)�l���/G�CD��                                    Bxz�x  ;          @����n{�1��!G����
CX� �n{��Q��c�
�)\)CE��                                    Bxz���  m          @����X���S33�ff��\)C`G��X�ÿ�p��j�H�/��CN�q                                    Bxz��d  "          @�\)�U��A��)����=qC^J=�U�����s33�:�CI޸                                    Bxz��
  
Z          @�ff�AG��@���5�C`�f�AG���p��}p��J  CJ)                                    Bxz���  �          @�
=�)���s33�����=qCk)�)������|���D�HCZ��                                    Bxz��V  "          @����*=q�x����
����Ck�H�*=q�  �z�H�Az�C\E                                    Bxz���  T          @���!G��z=q�ff�֏\Cm(��!G��  �~{�FQ�C]��                                    Bxz�ޢ  T          @�Q��G��a��
=q���Cdz��G�� ���g��0�HCTǮ                                    Bxz��H  �          @�\)�>�R�a��=q����Ce�H�>�R����u�=CTh�                                    Bxz���  T          @���K��c�
�������Cd=q�K��33�g
=�.�CT��                                    Bxz�
�  
�          @�ff�(������ff����Cn� �(��{�s�
�>��CaT{                                    Bxz�:  T          @��R������
=���
�fffC�33������p��.{� p�C}�                                    Bxz�'�  T          @��R�����(��u�,��C~������qG��O\)��
Czff                                    Bxz�6�  "          @�{����z��{���CvaH���!G��~{�P
=Cj=q                                    Bxz�E,  �          @�{��a��C�
�\)Clp�����������ep�CU��                                    Bxz�S�  ;          @��8�����_\)�,�
CZ��8�þ��R���\�HC:5�                                    Bxz�bx  �          @���z���ff�}p��333C����z��u��S33�!C��\                                    Bxz�q  T          @�{�Y�����\=���?�z�C�G��Y����\)�=q���C���                                    Bxz��  
�          @�(������
=����z�C��������33�5�\)C
=                                    Bxz��j  �          @������������R�HC~�q����fff�XQ��%�Cz�                                    Bxz��  "          @�33�#�
���H��Q���33C�AH�#�
�`���h���6Q�C�ٚ                                    Bxz���  
�          @��ͿL����=q�������RC�N�L���Z�H�o\)�;G�C�p�                                    Bxz��\  �          @��R�Ǯ���\����J=qC|#׿Ǯ�h���W��!�\Cv�)                                    Bxz��  �          @��R������{�G��
ffC}�H�����z=q�G
=���Cy�f                                    Bxz�ר  T          @��R�������k��#�C~�3�����u��N�R��HCz�H                                    Bxz��N  
Z          @����������Q���\)C�y�����aG��i���6��C�'�                                    Bxz���  "          @��Ϳ!G����H��33����C�C׿!G��X���tz��?�\C��{                                    Bxz��  T          @�ff�!G���ff���~�RC�Z�!G��fff�k��4��C�                                    Bxz�@  T          @��
��
=�����33��p�C�g���
=�B�\���\�T(�C�
=                                    Bxz� �  T          @�(��^�R��\)��Q����HC���^�R�Q��s33�@�C�                                    Bxz�/�  T          @��
���
��G���p���Q�C�ᾣ�
�Tz��w
=�E  C�C�                                    Bxz�>2  
�          @�ff=�Q���
=����ÅC��=�Q��C33��{�W�C��\                                    Bxz�L�  �          @�\)<��
��\)��p���Q�C�q<��
�fff�o\)�8\)C�(�                                    Bxz�[~  T          @�Q�=�G�����L(��  C��q=�G���\���\{C��                                     Bxz�j$  �          @�Q쾳33�������ٮC��쾳33�7���z��b�C�xR                                    Bxz�x�  �          @��
>�z��|(��g��*�C�3>�z������H��C��                                    Bxz��p  
�          @��?L���a��~�R�>�C�b�?L�Ϳ�{���R��C��                                    Bxz��  T          @�p�?J=q�dz��|(��<��C�C�?J=q��z���{�RC�33                                    Bxz���  T          @�(�?E��g��z�H�:��C��?E����H��{
=C�K�                                    Bxz��b  �          @�z�>u�Tz���{�M��C��>u�Q�����¡�C�1�                                    Bxz��  �          @��R>B�\�j�H�z�H�;�C�s3>B�\��G���
=�C�7
                                    Bxz�Ю  �          @��R?W
=�mp��w
=�5��C�XR?W
=������ffC�!H                                    Bxz��T  �          @�G�=�\)�n�R�}p��:��C��=�\)�����G�� C��                                     Bxz���  �          @���?^�R���H�\)��z�C�R?^�R�?\)�����^�C�                                      Bxz���  �          @�
=>��H���Ϳ�ff���C��q>��H�mp��y���8p�C��=                                    Bxz�F  �          @��R>���33�2�\��(�C��=>��'���ff�sG�C�b�                                    Bxz��  �          @��;W
=���H��33��33C�� �W
=�QG���G��K�RC�&f                                    Bxz�(�            @��=u���H�Q���p�C�Z�=u�A���ff�_  C���                                    Bxz�78  �          @����#�
��(��z����C�)�#�
�\(����\�M��C���                                    Bxz�E�  �          @�녾��
��
=��������C�AH���
�h������BC��H                                    Bxz�T�  �          @�{?=p���ff���
�Z�\C��?=p��x���j�H�+p�C�Z�                                    Bxz�c*  �          @��>���p���Q��x��C��3>��r�\�s33�4z�C��{                                    Bxz�q�  T          @���>�ff�����{��p�C���>�ff�i���{��;ffC��H                                    Bxz��v  T          @�\)�z�H��ff�'�����C�R�z�H�2�\��33�d��Cz��                                    Bxz��  T          @�
=��Q����ÿ�p���G�C��\��Q��aG��~�R�Az�C�                                    Bxz���  �          @�p�?p����(������h��C�5�?p���r�\�mp��-��C��)                                    Bxz��h  �          @��=#�
��(��\��ffC�<)=#�
�mp��vff�8z�C�S3                                    Bxz��  �          @�z�?Q���\)��\)��ffC���?Q��Z=q���\�EG�C�                                    Bxz�ɴ  �          @��?��\�����(��P��C���?��\�xQ��fff�'��C�^�                                    Bxz��Z  �          @��?�{���Ϳ�\)�?�C�R?�{�{��`���"�
C��                                    Bxz��   �          @�?�������
�\(�C�K�?���xQ��j�H�,=qC�c�                                    Bxz���  �          @�ff�#�
���ÿ��
�Z{C�"��#�
�}p��n{�,�C��R                                    Bxz�L  �          @��L������Y���  C���L������XQ��33C���                                    Bxz��  �          @���#�
��33�����p�C��#�
����J=q��HC���                                    Bxz�!�  �          @�z�h����\)�Q����C�Ϳh�������S33�G�C��=                                    Bxz�0>  
�          @�(��Y����Q�8Q���
=C�g��Y����
=�N{�G�C�L�                                    Bxz�>�  �          @���xQ����R�(���33C��{�xQ���\)�Fff�33C��                                    Bxz�M�  �          @���aG����
>Ǯ@�ffC��R�aG����
��
��z�C���                                    Bxz�\0  
�          @�zᾨ����33>�(�@���C�B��������
�G���\)C��                                    Bxz�j�  �          @�z��G����\?&ff@�\)C�b���G����R��
��Q�C�W
                                    Bxz�y|  �          @��
������G�?W
=A�
C�5þ�����Q������
C�)                                    Bxz��"  �          @�zῑ���\)��=q�:=qC��R�����p��4z���\)C��                                    Bxz���  �          @���\)��녾����J�HC��)�\)��
=�7��
=C�n                                    Bxz��n  �          @��
���R���H�L���
=qC�Uþ��R�����3�
����C��                                    Bxz��  �          @����  ��zᾊ=q�8��C�����  ����8���z�C�l�                                    Bxz�º  �          @����z���33��Q쿀  C�q쾔z�����-p���C�1�                                    Bxz��`  �          @�(��������
=u?��C�j=������ff�%����C�0�                                    Bxz��  �          @�(��W
=���
>W
=@�RC��)�W
=�����p����C��R                                    Bxz��  �          @�zᾮ{���
>#�
?�Q�C�4{��{��  � �����C���                                    Bxz��R  �          @�33�\���\>�z�@FffC����\��������ffC���                                    Bxz��  �          @��H�W
=���>�(�@��HC���W
=���\�\)����C���                                    Bxz��  T          @�(��Q�����>���@a�C����Q������z��̏\C�!H                                    Bxz�)D  �          @��H������>��@333C������\)�����C��3                                    Bxz�7�  �          @��H>8Q����>��H@���C��q>8Q����
�(���\)C�{                                    Bxz�F�  �          @��H>k�����?
=q@���C�=q>k���(������33C�XR                                    Bxz�U6  �          @��H=u����?#�
@��C�T{=u���33��Q�C�Z�                                    Bxz�c�  �          @�=q<��
��G�?
=q@���C�!H<��
���
�Q���G�C�#�                                    Bxz�r�  �          @�=q<����>��R@XQ�C�*=<��������z�C�/\                                    Bxz��(  �          @��׼���Q�>8Q�?�(�C�Ф����p����ۙ�C��=                                    Bxz���  �          @�\)�.{��
=�#�
�uC���.{�����$z���C���                                    Bxz��t  �          @�
=�aG���ff=L��?��C�Ǯ�aG����� ����Q�C���                                    Bxz��  �          @��׽���ff?(��@��C�Uý������p�����C�H�                                    Bxz���  �          @��׾B�\��ff?E�A{C��þB�\���Ϳ����  C��f                                    Bxz��f  
�          @�Q��
=���R>L��@
�HC��
��
=��z��������C�o\                                    Bxz��  �          @�Q�333��ff>��?˅C�.�333��33��H��{C���                                    Bxz��  �          @���(���(�?��@\C����(����������C�\)                                    Bxz��X  �          @�����p���z�?�Q�A���C��쾽p����R�5��{C��{                                    Bxz��  �          @�\)��z���33�����
C�e��z����!G���
=C�&f                                    Bxz��  �          @�  �W
=���;�����33C�` �W
=�����7��
=C�y�                                    Bxz�"J  �          @�z�O\)��\)?�@�(�C��)�O\)����
=����C�=q                                    Bxz�0�  �          @�������  ?(�@�p�C��\������(���\���C��                                    Bxz�?�  :          @�ff�u��G�?�Q�AK\)C��\�u��ff��ff����C��                                    Bxz�N<  T          @�z�Tz�����?&ff@�=qC�T{�Tz���ff������RC��                                    Bxz�\�  
Z          @�녿z�H���
?Y��A  C��{�z�H��(���\��=qC�Q�                                    Bxz�k�  "          @�����ff���?�
=Az�RC����ff������Q��Pz�C�"�                                    Bxz�z.  �          @�{������R?E�A
ffC�ÿ����ff��  ��  C��                                    Bxz���  �          @����G���p�@	��A�
=Cw)��G���{�.{���Cyk�                                    Bxz��z  �          @�녾��
�Mp�@��B[��C�!H���
����@��AΏ\C�+�                                    Bxz��   �          @�33���(Q�@�Bwp�C��������R@@��Bz�C�L�                                    Bxz���  
�          @�>���7
=@�z�Bcz�C���>����z�@(�A�33C���                                    Bxz��l  "          @�33�Ǯ��
@�(�Bq�HC�7
�Ǯ����@�RA�C�>�                                    Bxz��  �          @�G����
=@��\B��RC�Ff����ff@K�B�C�!H                                    Bxz��  �          @��
?�{�-p�@�{B\G�C�
?�{���@AمC��                                    Bxz��^  "          @��\?!G��<��@���BU\)C��q?!G�����@ffA�Q�C��=                                    Bxz��  T          @�=q���
�n{@W
=B(=qC��H���
��z�?�\)AN{C���                                    Bxz��  "          @���8Q��l��@XQ�B'�\C�xR�8Q���(�?�33AQC�Ǯ                                    Bxz�P  
�          @�\)��{�mp�=#�
?�RC�\)��{�P�׿�\��33C���                                    Bxz�)�  �          @��
?
=q�Z=q�Y���2(�C�z�?
=q������(��3C�                                    Bxz�8�  �          @�{?���l���>�R�\)C���?�녿��
����qC�H�                                    Bxz�GB  T          @�(�?�(��|���%�����C���?�(������p��k�HC��                                     Bxz�U�  �          @���?��R�q��   ��33C��?��R�������f{C���                                    Bxz�d�  "          @�=q?���r�\�!����C��?�������e�C�5�                                    Bxz�s4  "          @��?˅�n�R�$z����C���?˅� ������g
=C�+�                                    Bxz���  T          @�{?�G��q��)����C�~�?�G��G������d�RC���                                    