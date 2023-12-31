CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230401000000_e20230401235959_p20230402021714_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-04-02T02:17:14.401Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-04-01T00:00:00.000Z   time_coverage_end         2023-04-01T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxs��   "          @�z�@ �׿���@�(�B��)C�� @ ���i��@qG�B(ffC�g�                                    Bxs��  
�          @��H?B�\�=p�@�ffB�  C�H?B�\�`��@�\)BF�RC�                                      Bxs��L  
�          @�(�?B�\�h��@�  B���C��?B�\�l(�@�B@  C��{                                    Bxs���  T          @��=#�
��ff@�{B�33C��\=#�
�~�R@{�B2��C�N                                    Bxs��  
�          @����Ϳ��@�Q�B��\C���������Q�@\)B3p�C�AH                                    Bxs�>  
Z          @��R=�Q��ff@��B�ffC���=�Q���
=@uB)�C���                                    Bxs�)�  
�          @��>�33��z�@�z�B��C���>�33��Q�@l��B#�\C�U�                                    Bxs�8�  "          @��#�
���@��RB�\C��=�#�
���
@P��B��C�H                                    Bxs�G0  
�          @�p�?
=q��33@��B�.C��\?
=q��p�@`  B��C�|)                                    Bxs�U�  [          @��?   ����@�(�B��C�8R?   ��  @b�\B
=C�/\                                    Bxs�d|  
(          @��
>�Q��ff@�z�B�p�C���>�Q���G�@~{B*�\C�ff                                    Bxs�s"  T          @���?�(��G�@���B{p�C�1�?�(����\@Dz�B��C�}q                                    Bxs���  "          @�33?�=q�
=@���Bt(�C��{?�=q���H@:�HA�
=C��                                    Bxs��n  
Z          @�G�?�{�!G�@�=qBu�RC��?�{��=q@>{A�(�C��H                                    Bxs��  T          @��\?n{�?\)@�BfffC���?n{��z�@'
=AԸRC��                                    Bxs���  
Z          @���?�����=q@�(�B�.C�f?����g�@s33B3z�C�&f                                    Bxs��`  �          @�{?�?   @�
=B��A���?���@�ffBo�\C���                                    Bxs��  "          @�p�?��>�
=@��B��ATz�?���&ff@��Bf�HC�N                                    Bxs�٬  "          @�Q�?�����@�33B�{C��?���U@��B[\)C��\                                    Bxs��R  
(          @��?�(����@���B�
=C��?�(��o\)@�p�BM33C��                                    Bxs���  �          @ə�?p�׿G�@�B���C��?p���z�H@��HBI  C���                                    Bxs��  T          @��R?5�
=q@�z�B�\C�Y�?5�[�@�Q�BP�C��R                                    Bxs�D  T          @���?@  ��{@�(�B�L�C���?@  �y��@��RB:�\C�h�                                    Bxs�"�  �          @���?aG����@���B��)C��q?aG��y��@tz�B.�C�`                                     Bxs�1�  �          @1G���\)���?У�B��C�)��\)�.�R>k�@���C�H�                                    Bxs�@6            @B�\?   ��@�B.  C�Ǯ?   �5?#�
AG\)C��                                    Bxs�N�  
a          @u?xQ��9��@�BC�1�?xQ��j�H>�
=@ȣ�C�^�                                    Bxs�]�  
�          @tz�W
=�>{@G�Bp�C�  �W
=�n{>\@���C�g�                                    Bxs�l(  
�          @��<��
�H��@333B&�\C�0�<��
��z�?:�HA z�C�%                                    Bxs�z�  �          @�z᾽p��w�@p�B\)C�L;�p����\=�\)?Y��C��R                                    Bxs��t  	�          @��H=u��(��:�H�  C�c�=u�Vff�:�H�$Q�C���                                    Bxs��  	�          @�z�>�z���33�2�\����C���>�z��*�H��p��u\)C�
                                    Bxs���  �          @�{?������%��י�C�:�?��<(����\�h�RC�|)                                    Bxs��f  "          @��?
=q��  ��R��(�C��?
=q�=p���\)�f{C�#�                                    Bxs��  
�          @��\�xQ�����������C��ͿxQ��L�������S�C}(�                                    Bxs�Ҳ  
�          @����0�������
�H����C��0���J=q��
=�X��C�ٚ                                    Bxs��X  �          @��
��33�����33���C�\��33�Tz���{�T{C�                                      Bxs���            @����z���(��z���z�C�ff��z��J�H����^��C�h�                                    Bxs���  T          @���?��
����E��p�C�G�?��
�������|��C���                                    Bxs�J  T          @�ff������z��3�
��Q�C��\�����-p���ff�t33C���                                    Bxs��  �          @��
�����\)�>�R�  C�3�����R��Q��~�\C�=q                                    Bxs�*�  "          @�(�?��
��G����R�c�
C�޸?��
�{��0���  C���                                    Bxs�9<  T          @��R@����\)?��
A�ffC�h�@����  �O\)�(�C��q                                    Bxs�G�  	�          @���@���n�R@3�
BG�C�b�@������>�@�\)C��=                                    Bxs�V�  
(          @��
@#�
�s33@�A��C���@#�
���#�
�ǮC��                                    Bxs�e.  �          @��@#33�J=q@B�\BffC�w
@#33����?uA-�C�ff                                    Bxs�s�  
Z          @��@/\)���R?��RA�=qC��@/\)���
�h��� z�C��                                    Bxs��z  
�          @��
@�H���?��A8z�C���@�H������R���C�*=                                    Bxs��   
�          @�(�@'
=��33?��
A���C�w
@'
=���Ϳ=p���z�C���                                    Bxs���  
�          @�z�@3�
����?��A��\C��f@3�
�����Q��{C�*=                                    Bxs��l  �          @��@3�
���
?�{A��C�aH@3�
��ff�+��߮C�o\                                    Bxs��  "          @�Q�@5��|(�@%�A��C�޸@5���ff>#�
?��HC���                                    Bxs�˸  
�          @��R@#33��z�@(��AۮC��@#33��(����
�#�
C�=q                                    Bxs��^  
�          @��\?�������?��
A��\C�f?�����\)��=q�3�
C��)                                    Bxs��  T          @��H?Ǯ��Q�@��AУ�C��?Ǯ���H��Q��o\)C�"�                                    Bxs���  
�          @��R@���33@�z�BkffC���@��fff@P  B�C�                                      Bxs�P  T          @�{@!G����@�\)By��C���@!G��e@|(�B'�RC���                                    Bxs��  �          @�\)@�Ϳ�\@�
=B�p�C��@���aG�@�33BA�HC�o\                                    Bxs�#�  �          @�33@�\?L��@���B�ǮA���@�\�33@��RBh��C��                                    Bxs�2B  	�          @�ff?�G�?�@���B��fB7(�?�G���\)@���B���C��                                    Bxs�@�  T          @�@
=?��@���B��=A��@
=���@�Q�B}Q�C���                                    Bxs�O�  T          @ƸR@�?��@�33B�ffAt(�@��"�\@���Bi�
C�U�                                    Bxs�^4  T          @��@�=u@�{B�=q?��@��H��@��\BY�C�`                                     Bxs�l�  "          @�?����@�33B���C���?��W
=@��BL��C��
                                    Bxs�{�  
�          @�(�?�׾�=q@��HB�W
C��
?���W
=@��\BSp�C���                                    Bxs��&  
�          @��R?У׿��@�33B��qC��?У��l(�@���B3�C��=                                    Bxs���  "          @�33?�p��L��@���B�ffC��?�p��QG�@��BS��C���                                    Bxs��r  
�          @�  ?�����@�p�B��fC�,�?��P  @��BL{C�E                                    Bxs��  
�          @��
?��;��@�G�B���C��?����O\)@���BI�C��)                                    Bxs�ľ  
!          @��H@녾�@�p�B�\C�S3@��C�
@�{BC
=C�                                    Bxs��d  �          @�\)?�=q�u@��B�=qC�L�?�=q�\��@y��B7(�C�L�                                    Bxs��
  
�          @��@�>#�
@��B�(�@s�
@��#33@��\BS{C�~�                                    Bxs��  
Z          @��R@   <��
@�Q�B~  ?\)@   �"�\@�=qBI�
C�>�                                    Bxs��V  
�          @�Q�@?\)�aG�@��Bjz�C���@?\)�,��@��B6ffC���                                    Bxs��  �          @���@"�\�+�@��B|(�C��\@"�\�Mp�@��HB3�HC�"�                                    Bxs��  T          @�z�@�
>\)@��RB��f@s�
@�
��H@��HBW=qC�33                                    Bxs�+H  
�          @�=q?���>���@��B�.A]��?�����@��BnffC��=                                    Bxs�9�  �          @���@(�?��@�{Bn��B"�H@(��fff@��RB��C��H                                    Bxs�H�  �          @��?�p�@W�@�ffB<(�Bn�?�p�?B�\@��\B���A���                                    Bxs�W:  �          @��@   @C�
@�z�B@  BJ��@   >���@��HB�8RA��                                    Bxs�e�  "          @��@'�@ ��@��\BAz�B/z�@'�=�Q�@�G�Buz�?�z�                                    Bxs�t�  �          @�33@(��?�p�@s33BD=qB33@(�þk�@���Bh�HC�|)                                    Bxs��,  
�          @���@�@!G�@hQ�B9Q�B:@�>���@���Bu@��\                                    Bxs���  
�          @�p�@��@�@fffB<�B0��@��>8Q�@���Br��@���                                    Bxs��x  "          @���@'�@$z�@\(�B,�
B1��@'�>�@���BhQ�A (�                                    Bxs��  
�          @��@B�\@`��?�\)AX��BDz�@B�\@p�@/\)BQ�B�H                                    Bxs���  �          @�G�@N{@g
=?�G�Aj{BA�@N{@{@:�HB�
B�                                    Bxs��j  
�          @���@S�
@e�?���A\(�B=
=@S�
@\)@6ffB
  B�                                    Bxs��  T          @�Q�@\��@Z=q?�
=AZffB2��@\��@@/\)B33B                                    Bxs��  �          @���@n�R@E�?��Aw
=B��@n�R?�p�@-p�B��A�\)                                    Bxs��\  �          @��R@g�@8��?�p�A��
B�\@g�?�\)@<��BA��                                    Bxs�  T          @���@Z�H@�?���A�33B@Z�H?�z�@�HB��A���                                    Bxs��  �          @��H���R?�p���p��{Bř����R@tz��&ff��B�33                                    Bxs�$N  
Z          @�
==L��@1���ff�a�RB�
==L��@����{��\)B���                                    Bxs�2�  "          @�
=?��\@����R��33B�p�?��\@��H�L�Ϳ�RB�8R                                    Bxs�A�  T          @�33>�ff@�=q�(�����RB�8R>�ff@�녽�G���  B��                                    Bxs�P@  T          @���>���@���z��£�B��>���@��?�\@�Q�B�Ǯ                                    Bxs�^�  �          @��
?���@s�
����HB�=q?���@�33=�G�?���B�                                      Bxs�m�  "          @��\@	��@AG��*�H���BZ
=@	��@|�ͿJ=q��Bu��                                    Bxs�|2  "          @��?��@^{�"�\���B�ff?��@��þ�ff���B�                                    Bxs���  �          @��\@\)@hQ�J=q�"ffB^Q�@\)@b�\?��Al  B[z�                                    Bxs��~  �          @l(�?��@$z�?�=qA��BW33?��?�Q�@�RB%z�B'G�                                    Bxs��$  
�          @9��?�G�?�G�@
�HBH{B{?�G��W
=@Q�Bez�C��                                    Bxs���  �          @p�?\?�\?�z�B%�A��R?\���?�p�B/G�C�                                    Bxs��p  
�          ?�p�?����z�?�=qB4�C�{?�����?\(�Aڏ\C���                                    Bxs��  �          ?���?�p��z�?�{B4=qC�Y�?�p���Q�?c�
A�ffC�
=                                    Bxs��  �          ?˅?�G���33?�z�B>
=C�^�?�G��c�
?Tz�A��HC�T{                                    Bxs��b  "          ?�Q�?�33�0��?�33B9
=C���?�33���?^�RAՅC��                                    Bxs�   �          ?��R?fff�!G�?��
BY�
C�b�?fff���?��\BQ�C�&f                                    Bxs��  �          ?�?z᾽p�?���Bp�
C�� ?z�k�?Y��B�C�H                                    Bxs�T  "          @�H>�G�����?�BJ�C�Z�>�G��
�H?^�RA�=qC��=                                    Bxs�+�  
�          @�
=�#�
��33?�A��
C����#�
��33�5���C��q                                    Bxs�:�  T          @��\�����?�ffAM�C�Q��������
=��(�C�Ff                                    Bxs�IF  
Z          @�������=q?(��@�C������(���
���C���                                    Bxs�W�  "          @�p�>��
@�?��AO�B���>��
@n{@[�B*ffB�33                                    Bxs�f�  �          @��R?aG�@�Q�@Q�B�B�\?aG�@33@��\B�W
B�Q�                                    Bxs�u8  �          @��?��\@��@@  A��RB���?��\@$z�@��Bo�B~��                                    Bxs���  "          @�ff?�z�@vff@i��B!ffB~p�?�z�?��@��\B�ffB��                                    Bxs���  T          @��R?�\@o\)@z=qB-��B�k�?�\?���@�  B�k�B                                      Bxs��*  
Z          @�\)?�@_\)@��
B8�Bx��?�?�  @��B��
A�z�                                    Bxs���  �          @�
=?�Q�@N�R@�33BH(�ByQ�?�Q�?(��@��
B���A�p�                                    Bxs��v  "          @�
=@Q�@!�@���BR�B;  @Q�#�
@��RB��
C��=                                    Bxs��  
�          @�(�?��R@ ��@��B}�BVG�?��R�333@��\B��qC�S3                                    Bxs���  T          @��H?�ff@!G�@�ffBgG�BiQ�?�ff��@��HB�k�C�Ф                                    Bxs��h  �          @��?�{@1�@��
Bd33B�Q�?�{>��@�z�B�p�@�Q�                                    Bxs��  "          @�33?fff@U@��
BN��B���?fff?B�\@�{B��=B!(�                                    Bxs��  �          @��H@<��?J=q@�ffB^ffAp  @<�Ϳ�  @���BQ
=C��=                                    Bxs�Z  �          @��H@�>�=q@�{Bt�@ʏ\@���Q�@�  BP�\C��R                                    Bxs�%   	�          @�z�?�G�?aG�@��B�B�Aԏ\?�G����@�
=B{�C��{                                    Bxs�3�  
�          @�\)?�
=?(��@�
=B�u�A�z�?�
=��p�@�(�B}�RC��                                    Bxs�BL  
�          @��?s33��R@��
B���C�b�?s33�B�\@��BR(�C���                                    Bxs�P�  
�          @�33?J=q@g
=@Q�B&�\B�Q�?J=q?�ff@��
B��)B{�H                                    Bxs�_�  
�          @��?8Q�@�
=@%A�B��R?8Q�@7
=@��Bd�B��
                                    Bxs�n>  
�          @�p�?�G�@1�@�
=Bb(�B�.?�G�>�  @�  B��3A]                                    Bxs�|�  �          @��R?.{@y��@o\)B-��B�G�?.{?�=q@�p�B��B���                                    Bxs���  N          @��?E�@���@Y��Bp�B��?E�?�(�@��B�  B�G�                                    Bxs��0  (          @���?���@��?�G�A���B�\)?���@N{@s33B<z�B�ff                                    Bxs���  "          @��?��?���z��%�
B{��?��?˅��ff�lz�B���                                    Bxs��|  T          @/\)>�
=@�
?��RA��HB�>�
=?���@Bj��B�\                                    Bxs��"  "          @5?.{@�Ϳ=p���
=B��q?.{@�>�
=A=qB��                                    Bxs���  
Z          @z�H?L��@k�?@  A4  B�\)?L��@7
=@�Bz�B��{                                    Bxs��n  
�          @|��?L��@k�?�Q�A�=qB�z�?L��@(��@4z�B6G�B�aH                                    Bxs��  "          @��R?��@�Q�?�=qAnffB�W
?��@?\)@8Q�B-�RB��                                    Bxs� �  	�          @��\�W
=��ff�����
C�]q�W
=�(�ÿxQ��X��C|�=                                    Bxs�`  �          @�  ?����U����H��\)C��?����p��8���F33C���                                    Bxs�  �          @6ff?�p�?�
=������Be?�p�@��k���{Bx\)                                    Bxs�,�  �          @�=q?c�
@�z�?h��A.{B�33?c�
@hQ�@A�B��B�\)                                    Bxs�;R  �          @��\?G�@�=q?��At��B��R?G�@e@_\)B.33B���                                    Bxs�I�  �          @��
>�@���@33A�B���>�@Fff@���BPB��                                    Bxs�X�  �          @�Q�>���@��?�  A�=qB���>���@?\)@hQ�BI{B��H                                    Bxs�gD  �          @~�R?0��@w�<��
>aG�B��R?0��@X��?�\)A���B��                                    Bxs�u�  �          @��H?��R@s�
��������B���?��R@\)?#�
A33B��                                    Bxs���  �          @g�?�  @Z�H�8Q��6ffB�L�?�  @Vff?�  A�(�B���                                    Bxs��6  
�          @3�
?�ff@#�
>u@�  B�� ?�ff@��?�Q�A���Bp�                                    Bxs���  �          @P  ?��@3�
?aG�A}�B�#�?��@33@�B${Be��                                    Bxs���  
�          @b�\?��
@N�R?#�
A(  B�Ǯ?��
@!�@ffB=qB|��                                    Bxs��(  "          @XQ�?�(�@>�R?n{A�G�B�L�?�(�@
�H@\)B'�Brz�                                    Bxs���  �          @`  ?���@@��?���A�{B���?���@�@��B.�Bf
=                                    Bxs��t  T          @���?�@[�?��\A���B�.?�@=q@0  B,�\B]�                                    Bxs��  
Z          @��H@!�@���@33A�p�Bp\)@!�@1�@��B>ffB>�R                                    Bxs���  T          @�33@=p�@�G�@=qA�33Bn=q@=p�@R�\@�Q�B6  B@=q                                    Bxs�f  
�          @�(�@e�@�Q�@s33B�RBN  @e�@
�H@�\)BRp�A�p�                                    Bxs�  T          @��
@�  @fff@�ffB�HB(  @�  ?�Q�@���BQQ�A�=q                                    Bxs�%�  
�          @�\)@��H@X��@�  B ��Bz�@��H?\(�@��BTG�A>=q                                    Bxs�4X  
�          @�ff@s33@HQ�@��B2�\B��@s33>�@�Q�Ba�\@�
=                                    Bxs�B�  
Z          @�\@mp�@E�@�
=B=33B�R@mp�>�  @���Bi�
@y��                                    Bxs�Q�  
�          @�{@w
=@%�@�ffBFQ�B(�@w
=��z�@���Be(�C��q                                    Bxs�`J  	�          @�@�33@Q�@���B:�A��@�33��p�@���BS��C���                                    Bxs�n�  �          @�@�  ?��@�\)B?
=A��H@�  ����@��HBDQ�C�)                                    Bxs�}�  �          @��H@���8Q�@�\)B+��C�@���\)@�=qB�\C��                                    Bxs��<  
(          @�p�@����O\)@���B7��C��{@����<��@u�B  C��                                    Bxs���  "          @��@�=q����@��RB4=qC���@�=q�C�
@[�BC�T{                                    Bxs���  T          @��@Z=q�aG�@���BP��C��=@Z=q�8��@eB(�C��
                                    Bxs��.  "          @�G�@O\)��(�@�z�BN�C��@O\)�l(�@R�\Bz�C���                                    Bxs���  T          @�p�@[���{@N{A�  C�\@[����?(��@�{C���                                    Bxs��z  �          @�@G�����@FffAә�C�@G���
=>k�?��C�Q�                                    Bxs��   �          @�\)@Z=q�l(�@P  Bp�C�Z�@Z=q����?�33A2{C��                                     Bxs���  
�          @�G�@P���|(�@(Q�AٮC�Ǯ@P�����R>��H@�{C�Q�                                    Bxs�l  
�          @�  >����  @�p�B��C�b�>����=q@\)B1��C��                                    Bxs�  �          @�p�?
=q�'
=@���B���C��?
=q��z�@l��B��C�,�                                    Bxs��  T          @���<��
�c�
@���BU��C�/\<��
��(�@&ffA�z�C�                                      Bxs�-^  T          @�?�{�*�H@��Bl  C�xR?�{��@L(�BC�R                                    Bxs�<  �          @���@(���
=@��HBu��C�=q@(��vff@o\)B ��C��R                                    Bxs�J�  
�          @��@���  @�=qB�L�C��@��#�
@�(�BN�\C�'�                                    Bxs�YP  	�          @�@"�\    @�G�Bs�C���@"�\�ff@���BJ�C�9�                                    Bxs�g�  �          @���@G
=@.�R@�=qB8��B$��@G
=>�(�@��HBiQ�@�z�                                    Bxs�v�  
�          @��@N�R@K�@�(�B)\)B2\)@N�R?n{@�(�Bc33A�
                                    Bxs��B  �          @��H@<(�@�@�G�BO�HB�\@<(���@�=qBt\)C���                                    Bxs���  
`          @�z�@G
=?�G�@��RBY  A�R@G
=�5@�
=Bj\)C��\                                    Bxs���  "          @�\)@0��@�
@�=qBY��B��@0�׾\@�\)Bw��C��                                    Bxs��4  �          @��@ ��@a�@�G�B3  BZ�@ ��?�@�B|  A���                                    Bxs���  
�          @��
@(Q�@q�@�z�B.z�B\�\@(Q�?���@�(�By{A��
                                    Bxs�΀  "          @�\)@9��@&ff@��RBI��B'��@9��>#�
@�(�Bv�@O\)                                    Bxs��&  
�          @�p�@%?У�@��RBc�HB p�@%�333@�{Bu�RC�s3                                    Bxs���  �          @�Q�?��H���H@C33Br��C��?��H�p�@
=qB�
C��                                    Bxs��r  �          @���?\)��
=?��
A��C��?\)���R�J=q�
=C��=                                    Bxs�	  �          @�\)?.{��ff@C33Bz�C�T{?.{��33?0��@�\C��)                                    Bxs��  �          @��H?�ff�;�@�(�B[=qC��f?�ff��=q@AG�A��HC�9�                                    Bxs�&d  "          @���@33�
=q@�33Bs��C���@33��33@r�\B  C���                                    Bxs�5
  "          @ȣ�?����33@���B~{C��?������H@���B!��C��=                                    Bxs�C�  "          @�Q�@�R>�(�@�33B�#�A.=q@�R��@�(�Bo�
C���                                    Bxs�RV  �          @�p�@{?��@���B��qAt  @{� ��@�=qBrffC���                                    Bxs�`�  T          @�
=@h�ÿ�=q@�(�BE��C�AH@h���`  @Z�HBffC�\                                    Bxs�o�  T          @��?�Q�?�
=@�p�B�� B��?�Q��@�Q�B��C�9�                                    Bxs�~H  �          @љ�@�G�?��@x��BQ�A�z�@�G��u@��\B(\)C���                                    Bxs���  �          @�=q@l��@J=q@�B)p�B"  @l��?Tz�@�(�B[ffAJ=q                                    Bxs���  �          @��@HQ�@fff@�  B-\)BD(�@HQ�?���@�z�Bm(�A��
                                    Bxs��:  �          @�G�?�  @{�@J=qB�HB��3?�  @ff@��\Bw�HBl                                    Bxs���  �          @�p�>�{@�p�@C33B
{B�#�>�{@%@�ffBs�HB��                                    Bxs�ǆ  �          @�p�?��
@�33@AG�BQ�B�L�?��
@"�\@���Bm�B��q                                    Bxs��,  T          @�(�?�  @j=q@E�B  B���?�  ?�33@�z�BnG�B=z�                                    Bxs���  N          @�z�@��@Z�H?�33A���Bd�R@��@p�@N�RB7�HB4Q�                                    Bxs��x  Z          @��?��@j�H?�33A�G�B��?��@(�@U�BM��B�B�                                    Bxs�  �          @���?���@��@��
BffBl�?��;�Q�@��B��C���                                    Bxs��  
�          @�p�@(��@vff?�z�A�=qB^=q@(��@&ff@Z�HB*��B2G�                                    Bxs�j  
�          @��H@n�R@��R>�p�@j�HBH��@n�R@u@�
A�33B7\)                                    Bxs�.  
�          @�z�@n�R@�33?޸RA�p�B?  @n�R@:=q@XQ�B(�B�
                                    Bxs�<�  �          @��@���@y��?��AN�HB/��@���@;�@8Q�A�B\)                                    Bxs�K\  T          @�(�@I��@��@!G�AΏ\BS\)@I��@'�@�33B4�B                                    Bxs�Z  
�          @�@]p�@���?�=qA�33BO�@]p�@H��@eB�B)(�                                    Bxs�h�  (          @��@�  @dz�@ ��A�
=B��@�  @z�@W�BffA���                                    Bxs�wN  "          @���@��@�33?k�@���B-@��@n�R@6ffA�p�B                                      Bxs���  �          @��H@�Q�@���@�A��B
=@�Q�@,��@h��B\)A�\                                    Bxs���  �          @�z�@�
=@J=q@,(�Aʣ�B
=@�
=?�z�@s33B�
A��R                                    Bxs��@  
�          @��H@�33?�\@k�B�A���@�33���
@��\Bz�C��                                    Bxs���  �          @�Q�@�>��H@|(�B��@�
=@����
@p  B�HC�W
                                    Bxs���  "          @�33@�����G�@�G�B333C��@����G�@j�HB
=C�                                    Bxs��2  
�          @�ff@��\=��
@�(�B5Q�?���@��\��=q@~{B �C���                                    Bxs���  
�          @�Q�@��
�Tz�@�(�B2\)C��@��
�)��@e�B33C�]q                                    Bxs��~  
Z          @θR@L�Ϳ.{@���Bm�C��@L���A�@��RB;�C�N                                    Bxs��$  �          @�  @r�\��=q@���BX�
C�  @r�\�!�@�(�B5�HC�%                                    Bxs�	�  
Z          @У�@C33>�=q@�
=Bw=q@�\)@C33�\)@���BY(�C��)                                    Bxs�p  
�          @Å@N{�:�H@�=qBX�C��)@N{�(��@s33B)�RC�T{                                    Bxs�'  "          @�  @z�H�G�@�  B,�
C��3@z�H�x��@6ffAڣ�C��)                                    Bxs�5�  
�          @�
=@����S33@R�\A�
=C��@�������?��A^�\C��H                                    Bxs�Db  
�          @��H@�ff���H?���A�p�C�@�ff��  ��G��xQ�C��q                                    Bxs�S  T          @��
@y����?��AE��C��\@y�������=p��ᙚC�K�                                    Bxs�a�  
�          @��
@aG����H?��HA�  C�U�@aG���G�<#�
=�C��                                    Bxs�pT  �          @�G�@y���z�H?ٙ�A��HC�l�@y�����׾������C�33                                    Bxs�~�  T          @��@r�\��G�?�(�Ae�C���@r�\��  ��\��  C�
=                                    Bxs���  �          @���@fff��Q�?��AX��C�R@fff��p��(��ÅC���                                    Bxs��F  	�          @�G�@H����G�=��
?L��C�T{@H����z��\)����C��R                                    Bxs���  �          @�{@Vff��=q�8Q��Q�C���@Vff�fff��33���RC�~�                                    Bxs���  
�          @�33@}p��g
=�:�H����C��\@}p��=p��p���ffC��H                                    Bxs��8  �          @��\@1���  >Ǯ@���C���@1�����Ǯ���RC��H                                    Bxs���  
�          @��
@�����>�=q@7
=C�~�@���������C�]q                                    Bxs��  "          @�@HQ��j=q?�ffAEG�C�C�@HQ��p�׿����  C��                                    Bxs��*  
(          @�@hQ��1�@��AҸRC�<)@hQ��^�R?^�RA�C�{                                    Bxs��  "          @�@�33��33@4z�BG�C���@�33�{?��HA�Q�C�y�                                    Bxs�v  T          @�{@W��
=@1G�B��C�@W����H@��A��C���                                    Bxs�   "          @�����@�
=@  A��HB�.���@7�@tz�BM�B�p�                                    Bxs�.�  T          @�  ��(�@�\)?�  A<Q�B�  ��(�@h��@2�\B
=B�\                                    Bxs�=h  
�          @�  ��@��?�R@�\)B�=��@z�H@ ��A�B�{                                    Bxs�L  �          @�������@�G�@�
A�  B�  ����@I��@�  BHQ�Bۅ                                    Bxs�Z�  T          @�ff�E�@�=q@7
=A�p�B�8R�E�@<��@�Q�B_�
B�ff                                    Bxs�iZ  
�          @����(�@�  @K�B�RB����(�@!�@�p�Bt��B�B�                                    Bxs�x   �          @�
=��@��@N�RB�B�z��@'
=@�Q�BsQ�B�\)                                    Bxs���  T          @��\��(�@N�R@HQ�B/�\B�
=��(�?�\)@�{B�W
B�ff                                    Bxs��L  T          @���>u@!G�@9��BC�RB�33>u?�=q@l(�B���B��                                    Bxs���  "          @O\)?�p��#�
?�=qBG�C��?�p����?p��A�33C�                                    Bxs���  
�          @Vff@0  ���
>��
@�Q�C���@0  ��  ��ff��p�C��                                     Bxs��>  
(          @e@ ����
>�ff@�=qC�H�@ ����
��ff����C�J=                                    Bxs���  T          @z�H@��@��?�RAQ�C�ٚ@��@  �.{�#33C���                                    Bxs�ފ  �          @U?�
=� �׿W
=���C��?�
=���R������
C�5�                                    Bxs��0  
�          @Vff@�@�
�=p��V�\B%��@�@�>8Q�@H��B,\)                                    Bxs���  T          @#33?��?�Q�?��A��Bn�R?��?��?�  B3��BF�R                                    Bxs�
|  T          @Dz�@=q��  ?��\A�\)C��{@=q��Q�?=p�AmG�C���                                    Bxs�"  "          @1�?�33�˅?G�A�p�C�
?�33��G�=u?�  C���                                    Bxs�'�  "          @Z�H@   ����ff�G�C�@   ���H��33���
C��3                                    Bxs�6n  �          @y��@(��<�ͿL���<��C��f@(�����Q����
C��\                                    Bxs�E  
�          @P  ?�녾��
�.{��=qC�p�?�녾k���z��9�C�c�                                    Bxs�S�  �          @9��?�\@
=?�=qA�G�B��?�\?�{@�BQ�B��                                    Bxs�b`  
�          @�=q>�p�@Y��@   Bz�B�k�>�p�@@j�HBo�B���                                    Bxs�q  T          @~{?�?���@W
=Bo�B.(�?��\)@g�B���C�9�                                    Bxs��  "          @��H?���=u@qG�Bz?�z�?�����(�@^{B[�C�k�                                    Bxs��R  T          @�?����@�  B�z�C��\?��9��@W
=B8��C��                                    Bxs���  
�          @�G�?�
=?�(�@�=qB�Q�BM�?�
=����@�=qB���C��q                                    Bxs���  �          @��?��\?�G�@b�\B|p�B3\)?��\��\)@p  B��3C��                                    Bxs��D  T          @�
=@c33@L��?��AC�B({@c33@ ��@\)A�(�B�                                    Bxs���  
Z          @��H@c33@Tz�?���As\)B,(�@c33@   @#33A��B��                                    Bxs�א  �          @��R@Dz�?�=q@Q�B533A���@Dzᾮ{@\(�B@ffC�ٚ                                    Bxs��6  �          @��R@tz�?�{@'�BQ�A��H@tz�>�
=@C33B@�Q�                                    Bxs���  T          @�\)@p��?p��@G�B�Aa�@p�׾Ǯ@O\)B"ffC�                                    Bxs��  �          @�@!�?�Q�@/\)B-{A��@!�>aG�@E�BJ=q@�
=                                    Bxs�(  T          @�Q�@J�H�C�
@,��A�=qC���@J�H�x��?�  A`  C��3                                    Bxs� �  T          @�
=@!��:�H@*=qB	�HC�t{@!��o\)?��\AzffC��                                    Bxs�/t  �          @�G�@@����\@7
=B=qC�XR@@���O\)?��HA��\C�l�                                    Bxs�>  �          @��@L�Ϳ�p�@G
=B(��C�y�@L����@
=A�\)C���                                    Bxs�L�  	�          @���@=p��
=@UB?Q�C�T{@=p���\)@4z�Bp�C���                                    Bxs�[f  
�          @�
=@Fff?Q�@�33BO�HAlQ�@Fff�aG�@��HBO  C�R                                    Bxs�j  
�          @�p�@�@	��@l��BE�\B*G�@�?   @�  Br�\AB�\                                    Bxs�x�  "          @�ff@z�@N�R@_\)B%=qBY(�@z�?�=q@�\)Bg��B	{                                    Bxs��X  �          @���@C�
@.{@R�\B��B&z�@C�
?���@�33BL�A�G�                                    Bxs���  
�          @��@4z�@�@s33B7p�B \)@4z�?.{@�Bc
=AY�                                    Bxs���  �          @��Ϳ�R@�  ?��\AO33B��)��R@���@UB�HBÀ                                     Bxs��J  
Z          @���(�@���=�G�?�=qB�#��(�@�@�RA�p�B���                                    Bxs���  
�          @�����@�33?   @��B��ÿ���@��R@)��AՅB�B�                                    Bxs�Ж  "          @���'
=@�(�?W
=Ap�B��H�'
=@�z�@2�\A�
=B�k�                                    Bxs��<  T          @�z���@��\?h��A{B�33��@���@:�HA�  B�8R                                    Bxs���  T          @�p��.{@���?��A!�B��.{@��\@>{A���B�8R                                    Bxs���  �          @�{��@���?�  AC33B�#���@�\)@P��B�\B�Ǯ                                    Bxs�.  
�          @�p��3�
@�=q?W
=A	G�B�u��3�
@��
@)��A�=qB��                                    Bxs��  �          @����8Q�@��?�ffA,Q�B� �8Q�@}p�@3�
A�G�B�                                    Bxs�(z  "          @�z�� ��@��\?�p�AJ{B�.� ��@��R@FffBp�B�{                                    Bxs�7   
Z          @�=q����@��?��
A���B�uþ���@�z�@j�HB%��B�.                                    Bxs�E�  �          @�      @�(�?��HA���B���    @~{@r�\B.z�B���                                    Bxs�Tl  T          @�ff=�\)@��
?�A�(�B�z�=�\)@�Q�@j�HB)�B�\                                    Bxs�c  "          @��?��H@��=u?=p�B���?��H@�33?��A���B��                                    Bxs�q�  �          @�Q�?���@�Q�>�@�G�B���?���@�@��A��HB�#�                                    Bxs��^  �          @�ff?���@�Q�?�A�
=B��?���@p  @W�B��B�33                                    Bxs��  �          @�=q@,��@y��@#�
A��B]�@,��@'�@w
=B6��B0z�                                    Bxs���  
U          @�Q�?�@���@"�\A�=qB�{?�@]p�@�B;ffBx(�                                    Bxs��P  '          @���@=q@�z�@J�HB(�BoG�@=q@'
=@�Q�BO=qB=Q�                                    Bxs���  �          @��@)��@N�R@x��B+BJ=q@)��?�  @�=qBf�HA�                                    Bxs�ɜ            @��R?��H@\��@@��B
=B�8R?��H@�\@��HBi�BZ�                                    Bxs��B  
�          @�?��ÿ�@@  B��RC�H�?��ÿ�@#�
BH�
C�,�                                    Bxs���  
�          @�  ?�{�mp�@�A�=qC�Q�?�{��33?Tz�A�
C��3                                    Bxs���  T          @�\)@&ff>u@��\Bkz�@��R@&ff��@��HBXffC���                                    Bxs�4  
�          @�G�@-p�?�@9��B&  B@-p�?(�@XQ�BJ\)AJff                                    Bxs��  
Z          @�G�@:=q@,��@$z�B�B+�@:=q?��
@X��B7�\A�p�                                    Bxs�!�  �          @�ff�\)@)��@�
=Be=qB�  �\)?\(�@��B�L�B�B�                                    Bxs�0&  T          @��\@"�\@n{?��RA�=qB^�R@"�\@,��@O\)B$��B;                                      Bxs�>�  �          @�  @L(�@xQ�?���Ap��BJz�@L(�@E@-p�A�G�B0z�                                    Bxs�Mr  "          @���@`��@g�?�=qA���B7@`��@1G�@4z�B{B�                                    Bxs�\  "          @�\)@I��@hQ�?�z�A��BD33@I��@)��@HQ�B�B G�                                    Bxs�j�  	�          @��@/\)@Fff@1�Bz�BA�@/\)?�@n�RBB{B33                                    Bxs�yd  
�          @�Q�@@!�@g
=B9(�B<��@?�  @�G�Bm=qA���                                    Bxs��
  
Z          @�  @@   @���BL�BH�\@?Q�@�p�B���A�                                    Bxs���  "          @�G�@{@k�@9��B�B`=q@{@@��BH  B-p�                                    Bxs��V  �          @��H?�
=@C�
@n�RB?ffB��)?�
=?��H@�33B�#�B6                                      Bxs���  �          @��@
�H@\��@XQ�B��Bg(�@
�H?���@�p�Bb=qB'��                                    Bxs�¢  "          @��?�z�@>�R@��BL��Bs�?�z�?���@�G�B�B�H                                    Bxs��H  
(          @��?�p�@*=q@�z�Bb��Bs�R?�p�?B�\@�G�B�#�A�\)                                    Bxs���  
�          @�  ?�p�@p�@�BZ�BLz�?�p�?&ff@���B��fA�                                    Bxs��  
�          @��?��R�#�
@X��B�ǮC�e?��R��@:�HBK�\C���                                    Bxs��:  
�          @�G�?��H�!G�@j=qBI{C��?��H�l(�@�RA�33C�g�                                    Bxs��  
�          @���?����@�G�B�� C���?���>�R@l��B?G�C���                                    Bxs��  '          @�G�?�p���@��\Bu�RC��?�p��{�@h��B&  C���                                    Bxs�),  "          @�z�?��
�(��@���Bt��C���?��
���H@u�B"�\C��                                    Bxs�7�  �          @\?G��A�@�p�Bk33C�@ ?G����R@mp�BffC���                                    Bxs�Fx  T          @�Q�?L���b�\@���BI�C�e?L������@3�
A�(�C���                                    Bxs�U  
�          @�\)@���@Mp�?�(�AF�RB�@���@"�\@33A�G�A�ff                                    Bxs�c�  �          @���@5@L(�@�RA��HBA33@5@z�@^{B2G�B�                                    Bxs�rj  �          @��@�
@�  @%�A��
Bo�@�
@2�\@w
=B;�BI��                                    Bxs��  T          @���?���@�@EB�RB��?���@AG�@�\)BXQ�B��{                                    Bxs���  "          @���?��@�  @"�\A�B��q?��@Q�@\)BDG�B�8R                                    Bxs��\  �          @�Q�?�ff@�  @2�\B  B�L�?�ff@=p�@�(�BS
=B�                                    Bxs��  
�          @�=q?�=q@��R@�RA�z�B���?�=q@QG�@z�HB?�B�                                    Bxs���  T          @�p�?�\@�@�A�{B�� ?�\@S33@p��B4�\BwQ�                                    Bxs��N  
�          @�=q@33@��\@�A�33B�B�@33@N�R@k�B/ffBf(�                                    Bxs���  
�          @��R?�z�@�  ?�\A�Q�B��?�z�@�G�@\(�B\)B�                                    Bxs��  �          @�Q�?�=q@���?�G�A�\)B�G�?�=q@�=q@[�B��B��3                                    Bxs��@  "          @��H@.�R@O\)@p�A�{BG��@.�R@\)@N{B)Q�B�                                    Bxs��  
�          @�{@c33@E@
�HAř�B${@c33@�@G�B=qA�z�                                    Bxs��  T          @�  @dz�@^{?�Q�A��B0@dz�@)��@3�
B{B33                                    Bxs�"2  
�          @�G�@mp�@N�R?��HA���B$33@mp�@z�@>{BB ff                                    Bxs�0�  
�          @��
@w
=@HQ�?��HAZffB=q@w
=@   @\)A�p�B�
                                    Bxs�?~  '          @��@i��@a�    <��
B/�@i��@S33?�  Ac\)B(G�                                    Bxs�N$  
�          @�z�@��@�{������Q�Bvp�@��@�G�?�Q�AY�Bs=q                                    Bxs�\�  "          @��@5@g��p���8��BO��@5@n{>Ǯ@�=qBR��                                    Bxs�kp  �          @�Q�@G
=@^{?��AIB@��@G
=@8��@��A���B+p�                                    Bxs�z  
�          @��\@N�R@S�
@#�
A�B7�@N�R@p�@c�
B)G�B	�\                                    Bxs���  �          @���@AG�@S33@:�HB�B=��@AG�@z�@x��B:�
B	�                                    Bxs��b  T          @��@{@-p�@vffB>�HBJ@{?�Q�@���Bt33A�ff                                    Bxs��  �          @�\)?�\?��@�33B���A�33?�\����@�  B�L�C���                                    Bxs���  �          @�Q�?.{=�G�@���B��3Aff?.{��Q�@��
B�C�f                                    Bxs��T  �          @���?\(�@0��@J=qB=��B�k�?\(�?��R@z�HB�� BpQ�                                    Bxs���  �          @���?z�H@�33@	��A�ffB�8R?z�H@Fff@\(�B:�B�\                                    Bxs��  
�          @��H>��R@w�@�A�p�B��
>��R@3�
@dz�BN\)B�ff                                    Bxs��F  "          @��>k�@j=q@��A�33B���>k�@)��@XQ�BO33B�{                                    Bxs���  �          @�  ?��H@*=q@E�B:(�B�
=?��H?�
=@s�
B(�BG=q                                    Bxs��  "          @�
=?�
=@<(�@n�RBF��B�#�?�
=?�(�@���B���BL��                                    Bxs�8  
Z          @�ff?�=q@:�H@g
=B=�Bv  ?�=q?�  @�z�B~G�B-��                                    Bxs�)�  "          @��R?z�@(�@Mp�BO�
B�k�?z�?�Q�@vffB���B�u�                                    Bxs�8�  
Z          @�p���?�=q@^{B{G�C�῵�.{@hQ�B�
=C:��                                    Bxs�G*  	�          @��H�
=?��@EBO��C&��
=�
=q@EBOC@��                                    Bxs�U�  �          @h��>��>��?�  B���B6Q�>������?��
B�B�C�Ǯ                                    Bxs�dv  
Z          @���@E@6ff>��@�
B*�H@E@(Q�?�{Aup�B!�                                    Bxs�s  
�          @��R@*=q@`��>�(�@�G�BS��@*=q@J=q?���A��BG                                    Bxs���  �          @��@8��@g�>B�\@
=BM��@8��@U?�z�A�p�BD��                                    Bxs��h  
�          @��@/\)@mp�>�p�@��
BU��@/\)@W
=?�{A�z�BK�                                    Bxs��  
�          @�ff@�z�?��@�RAˮA��@�z�?J=q@)��A��A"�\                                    Bxs���  "          @��\@}p�@  @��A�p�A�33@}p�?�33@1�B�HA��                                    Bxs��Z  �          @�ff@Fff@@A�B
=B@Fff?s33@c33B>z�A��                                    Bxs��   �          @�G�@�>8Q�@mp�Bp@�33@���=q@c�
B`��C��                                    Bxs�٦  T          @�=q@z�@N{?�@�z�Be
=@z�@6ff?�=qA���BX33                                    Bxs��L  "          @�ff?�z�@��?aG�A z�B��H?�z�@�33@��A���B�                                      Bxs���  �          @��@z�@�G�?��RA�G�Bp�\@z�@H��@N�RBQ�BV=q                                    Bxs��  
�          @��@'
=@��?�R@�p�Bq�R@'
=@�z�@�A�p�Bf�H                                    Bxs�>  T          @��?���@�{?���AE�B���?���@�G�@-p�A�(�B�W
                                    Bxs�"�  �          @�Q�?��H@���@7�B��B�=q?��H@=p�@��HBLp�B~��                                    Bxs�1�  �          @��=�G�@��R@>{BffB�=q=�G�@N�R@���BS�
B�.                                    Bxs�@0  
�          @���@333@@g�B3  B�\@333?�G�@�{BZ\)A���                                    Bxs�N�  X          @��@1�@(�@w�B>(�B�R@1�?J=q@��
Bb�A|��                                    Bxs�]|  �          @�
=@%?xQ�@��
BfA�@%��@�{Bm{C�9�                                    Bxs�l"  T          @��R@6ff?���@��BW33A�Q�@6ff��Q�@���B`p�C�c�                                    Bxs�z�  
�          @�  @   ?��R@��\B]z�B3Q�@   ?
=q@�Q�B��Ao�
                                    Bxs��n  "          @��@�\@ff@�=qB_�\B733@�\?\)@���B�#�Aw\)                                    Bxs��  "          @�33?�(�@�@�B^�\B:=q?�(�?
=@�(�B�.A�(�                                    Bxs���  �          @��@(����
@�=qB��)C��)@(���G�@��Bi\)C��=                                    Bxs��`  T          @�\)@z�?��@�=qBt��AW�@z�c�
@���Bo\)C���                                    Bxs��  
�          @�Q�@   >��R@vffBc
=@ᙚ@   �u@o\)BYp�C��                                     Bxs�Ҭ  T          @�z�@{�Q�@�G�Bd�C��@{�@c33B>�
C��                                     Bxs��R  T          @���@{��  @l��BT�RC��
@{�Q�@FffB(�C�f                                    Bxs���  
�          @��R@.�R?L��@l(�BQG�A�33@.�R��(�@p  BVQ�C�~�                                    Bxs���  
�          @���@���?�Q�@*=qB ��A�\)@���>�=q@9��B=q@s�
                                    Bxs�D  T          @�@��?�{@   A�  A��@��?u@>{B�AQ��                                    Bxs��  "          @�Q�@��H?�p�@7�B��A���@��H>�@L(�B33@ָR                                    Bxs�*�  T          @�Q�@�Q�?�(�@.{A��RAљ�@�Q�?}p�@Mp�Bp�A^�\                                    Bxs�96  �          @�(�@\)>�ff@�Bu33A5G�@\)�k�@�33Bm�RC��                                    Bxs�G�  T          @�G�?�33>�@��\B��@�ff?�33��@��B��3C��=                                    Bxs�V�  "          @��H?��H?��@��B�B$33?��H��R@�(�B�L�C���                                    Bxs�e(  
�          @���?��?Y��@�Q�B�
=A��H?�녿L��@�Q�B���C��                                    Bxs�s�  �          @��
?��?z�H@�
=B��HA��?�녿�@�G�B��C�g�                                    Bxs��t  
�          @�Q�?�G�=L��@�z�B�� ?��?�G���@�p�B{�\C��                                     Bxs��  �          @��H?��R�aG�@���B�8RC�� ?��R��33@�\)By
=C��                                    Bxs���  
�          @�@$z�?(�@��\Bj��AU�@$z�J=q@���Bh  C�e                                    Bxs��f  
(          @�p�@)��?0��@�  Bd�HAi�@)���0��@�  Bd�HC���                                    Bxs��  
�          @��\@N{?
=q@��HBM�A�@N{�G�@���BJ�C�,�                                    Bxs�˲  
�          @��@A�@G�@u�B9�B�@A�?5@���BW�HAS\)                                    Bxs��X  "          @��@=q@E@aG�B'�BP\)@=q?�=q@�=qB\{B                                    Bxs���  
�          @�=q@*�H@)��@l(�B1�HB2��@*�H?���@��HB]z�A��                                    Bxs���  
�          @�p�@8Q�@�R@uB5Q�B"�R@8Q�?�33@�B[�A��                                    Bxs�J  
�          @���@<(�@XQ�@Mp�B\)BD=q@<(�@p�@�33B@��B                                      Bxs��            @���@,(�@P  @VffB�
BI=q@,(�@�\@�{BL��Bp�                                    Bxs�#�  &          @���@ff@7�@hQ�B1��BJ@ff?���@�33BcQ�B�                                    Bxs�2<  	�          @���@�@@��@j=qB/Q�BP�R@�?�(�@��Bb��B��                                    Bxs�@�  
Z          @�G�@   @)��@�33BK��BT{@   ?�  @�
=B}�HB {                                    Bxs�O�  
�          @���?���@1G�@�BR��Bq�
?���?��@��\B���B"                                      Bxs�^.  
�          @���?��R@ ��@�33Bc�RB5(�?��R?�@�Q�B��A
=                                    Bxs�l�  �          @�@p�?�@�Bm(�B@p����
@���B�{C�˅                                    Bxs�{z  �          @�{@B�\>8Q�@���BV��@W
=@B�\��{@�Q�BL=qC���                                    Bxs��   �          @��\@i��>�@X��B+z�@33@i���k�@QG�B#�
C��\                                    Bxs���  �          @��\?�ff?�z�@�  B��RB?  ?�ff��\)@�z�B���C�~�                                    Bxs��l  "          @�G�?�Q�?fff@�  B�(�B(�?�Q���@���B��C��
                                    Bxs��  
�          @�=q@G�?8Q�@�
=B��3A��
@G��0��@�\)B��C���                                    Bxs�ĸ  
Z          @�33@Q�?aG�@�(�Bo��A��@Q�   @�{Bu
=C��                                    Bxs��^            @��\?�?�\)@���B}G�B=q?���\)@�\)B���C��3                                    Bxs��  
�          @�G�?�{?�G�@��Bv  BG�?�{=���@��B�p�@<��                                    Bxs��  T          @��?�z�?�G�@���B��)Bb�
?�z�>�{@�\)B�ffA�p�                                    Bxs��P  
�          @��?O\)>�p�@<��B�Q�A�p�?O\)���@:�HB��fC��                                    Bxs��  
�          @��H@z�.{@P  B`�\C��
@z�У�@8��B>��C��                                    Bxs��  T          @��?��H��
=@���Bz��C��?��H�4z�@U�B=G�C��                                    Bxs�+B            @�Q�?�
=�&ff@P��B>�RC�]q?�
=�]p�@z�A�ffC�0�                                    Bxs�9�  
�          @��\@z�?��
@n{BZ{A�=q@z�=���@{�Bm�\@��                                    Bxs�H�  
�          @�(�@#33>�@QG�BN�\A$(�@#33�
=q@P��BMp�C��)                                    Bxs�W4  �          @�?aG�?��@�B��BG��?aG����@���B�  C��R                                    Bxs�e�  �          @��?�z�?�Q�@��B�  B ?�zὣ�
@�\)B�  C�t{                                    Bxs�t�  
(          @��?ٙ�?��@���B���A��?ٙ����@�z�B�.C���                                    Bxs��&  �          @�(�?�
=?ٙ�@�Q�By�HBH=q?�
=>�33@��\B�ffA_33                                    Bxs���  �          @���?��?�ff@��\B��qBp?����(�@�p�B��)C�@                                     Bxs��r  
(          @�Q�c�
?s33@��B�CͿc�
�
=@��B�W
CU�\                                    Bxs��  �          @�녿\(�?G�@���B��=C	�R�\(��G�@���B��{C^33                                    Bxs���  "          @�=q>���?�\)@�z�B�Q�B��>��;�@�=qB�z�C���                                    Bxs��d  �          @�Q�@x��?�33?�A��A��@x��?�  @��A���A��                                    Bxs��
  
�          @�p�@L(�@0��@>{BB#�R@L(�?�G�@j=qB4�A癚                                    Bxs��  �          @���@>{@@��@0  B�B5�R@>{@z�@aG�B0�B�                                    Bxs��V  �          @�{@5@+�@>{B��B-G�@5?�Q�@g�B>p�A��
                                    Bxs��  �          @�
=@'
=@?\)@?\)B�HBC�@'
=?�p�@o\)BB�HB��                                    Bxs��  T          @�=q@   @W
=@2�\B
=Bm{@   @��@j=qBF  BH�R                                    Bxs�$H  	�          @��H@ff@J�H@=p�BffBb{@ff@
�H@p��BL�
B7��                                    Bxs�2�  �          @�G�@@j�H@��A�G�Bqff@@6ff@N�RB)�
BW                                      Bxs�A�  
�          @�(�@�@z=q@�A��Bzff@�@I��@E�B��Bd�                                    Bxs�P:  T          @�p�@�\@�=q?�A�=qB}p�@�\@W
=@<��Bz�Bj                                    Bxs�^�  �          @�33@C33@*�H?�\)A��
B$��@C33@ ��@#�
B33B�
                                    Bxs�m�  �          @�{@^�R��@*�HBffC�XR@^�R����@��B�RC��q                                    Bxs�|,  "          @�{@:�H?k�@dz�BE��A�Q�@:�H���@k�BNQ�C��                                    Bxs���  �          @�33@��@��@n�RBH�B-Q�@��?xQ�@�{Bn�A��                                    Bxs��x  "          @�  @Dz�?�ff@}p�BG�RA��@Dz�=���@�p�BVz�?��                                    Bxs��  �          @�=q@<(�?�G�@�  BEA�z�@<(�?�@��HB]�RA"�\                                    Bxs���  
�          @��H@<(�@�@s33B6z�B33@<(�?��@���BW�A���                                    Bxs��j  "          @�33@.�R@:�H@a�B%B;�\@.�R?��
@�
=BQ(�Bz�                                    Bxs��  �          @���@K�@,(�@P��B=qB �
@K�?��@x��B=�A�ff                                    Bxs��  "          @�33@W
=@!�@N{B��B
=@W
=?�  @s�
B8  A�
=                                    Bxs��\  
�          @�p�@��@��@
=qA�p�A�(�@��?�Q�@+�A��\A�(�                                    Bxs�   
Z          @�ff@��H@
=q@�A�  A��H@��H?�z�@6ffB�\A��                                    Bxs��  �          @�Q�@�=q@��?@  A
�\A�\)@�=q@z�?�33A�Q�A�33                                    Bxs�N  
Z          @��H@�p�@�?��Aep�A�(�@�p�?�\)?�A���A�p�                                    Bxs�+�  
�          @�G�@�G�@
=?�\A���Aљ�@�G�?\@33A�ffA�=q                                    Bxs�:�  �          @�Q�@g�?��
@*�HB�AѮ@g�?s33@Dz�B33Al                                      Bxs�I@  
�          @��@��@��
?��
A�  B�8R@��@{�@@��BBu�                                    Bxs�W�  
�          @�{?�z�@�(�?���A�=qB��?�z�@z�H@E�B
=B�                                      Bxs�f�  �          @�33@X��@-p�@Y��B33BQ�@X��?��@���B;z�A�ff                                    Bxs�u2  
�          @�z�@e�>u@�=qBH��@y��@e��z�H@��RBB\)C�Y�                                    Bxs���  
�          @��@g
=�aG�@�{BKffC�C�@g
=���H@�ffB<�RC��                                    Bxs��~  �          @��@����B�\@y��B/��C���@������
@l(�B$=qC�>�                                    Bxs��$  �          @�  @@��>�z�@�  B`�@��R@@�׿z�H@���BY(�C��                                    Bxs���  
�          @��R@j=q��ff@4z�B�C�B�@j=q��@ffA�{C�|)                                    Bxs��p  T          @�\)@�Q쿝p�@%A�ffC��R@�Q��33@��Ař�C���                                    Bxs��  �          @�@y���QG�?(��@�C��@y���Tzᾏ\)�G
=C���                                    Bxs�ۼ  �          @���@s�
�>{?���A^ffC�@s�
�K�>��
@n�RC�3                                    Bxs��b  T          @�33@G��W
=@ ��A��C�j=@G��r�\?�  A8Q�C��R                                    Bxs��  �          @�33@��ÿ�G�?�(�A��RC���@��ÿ���?�\)A��HC�'�                                    Bxs��  T          @�(�@�(����?�A��C��=@�(���
=?�  A733C�H                                    Bxs�T  �          @���@�����R?�{A�(�C�33@���O\)?��HA��C�L�                                    Bxs�$�  �          @��H@���>�z�?У�A���@Tz�@����.{?�33A��\C��                                    Bxs�3�  
�          @���@�33?B�\?��A9�A�@�33>�G�?�p�AW�
@��                                    Bxs�BF  
�          @���@�{?�R?\)@���@�Q�@�{>�G�?5@���@��                                    Bxs�P�  T          @���@�{?Q�=���?��A33@�{?B�\>��
@]p�AG�                                    Bxs�_�  
�          @���@�?�������fffA5p�@�?������p�A>{                                    Bxs�n8  T          @�p�@�z�?}p����H�Z=qA6�\@�z�?��
�c�
� ��Aj�R                                    Bxs�|�  T          @��\@�p�?����\����A��
@�p�?�������ffA��\                                    Bxs���  T          @�=q@�Q�?#�
�@���33A@�Q�?�Q��.{�=qA�                                      Bxs��*  T          @�
=@]p���{�W��0\)C�+�@]p�?
=�U��.(�A{                                    Bxs���  �          @�Q�@[�>u�J=q�*��@�G�@[�?���>�R��A��                                    Bxs��v  
�          @�@@  ��R�`  �C\)C�(�@@  >�33�b�\�F�@׮                                    Bxs��  T          @��@H�ÿ(��L(��3C�u�@H��>�=q�O\)�7(�@��                                    Bxs���  T          @��@AG������Z�H�:�
C���@AG��#�
�e�G�
C�~�                                    Bxs��h  
�          @��@Fff�����R�\�-��C��\@Fff��R�e�B�C�Q�                                    Bxs��  "          @�  @�녿��R�7
=��G�C�/\@�녿�33�R�\�{C��                                    Bxs� �  �          @��
@��\�   �4z���Q�C�G�@��\����X���z�C�
=                                    Bxs�Z  "          @�ff@��R�4z��\��{C�)@��R�\)�p��ڏ\C��                                    Bxs�   "          @�@|(��9�����z�C���@|(��
�H�A��{C���                                    Bxs�,�  
�          @�ff@|(��N{�!��ӮC�e@|(���H�S33�{C�:�                                    Bxs�;L  
Z          @�33@�G��HQ��  ��=qC�%@�G��=q�@���G�C���                                    Bxs�I�  �          @�\)@�\)�<�Ϳ�{��z�C��=@�\)�ff�%����\C�|)                                    Bxs�X�  
�          @�=q@���=p���\)����C���@���ff�%�ޣ�C���                                    Bxs�g>  
�          @��
@y���^{��33�G�C�0�@y���A���\���C��                                    Bxs�u�  "          @�z�@{��b�\�xQ��&=qC���@{��J=q������C��)                                    Bxs���  �          @��@^�R����@  ���C���@^�R�x�ÿ������
C��                                    Bxs��0  "          @���@U������  �P��C�{@U��p�������C���                                    Bxs���  "          @�  @ff��33��ff���
C��H@ff�}p��=p���HC�S3                                    Bxs��|  "          @���@*=q��Q�@  ��=qC��
@*=q��z��p�����C��R                                    Bxs��"  "          @��@\)���
=q��=qC�ff@\)���
������\)C�/\                                    Bxs���  T          @��@-p������(���ۅC��)@-p���ff��z�����C��=                                    Bxs��n  	�          @��R>����Q����
�~��C���>��Ϳ�G���=q�C���                                    Bxs��  
�          @�?
=q�'
=��ff�q�RC��?
=q���
��
=(�C�XR                                    Bxs���  �          @��?(���5������effC���?(�ÿ�����C���                                    Bxs�`  
(          @�z�>��;���\)�b
=C���>���33��33C�
                                    Bxs�  �          @��\@��`  �Tz��G�C��=@�� �����
�Lp�C�q�                                    Bxs�%�  "          @���@c33�DzῺ�H���\C��{@c33�%��{��Q�C�H                                    Bxs�4R  �          @��H@�G��C�
��Q��z=qC�@ @�G��7
=��33�H��C�'�                                    Bxs�B�  �          @���@�=q�L�ͽ���{C��@�=q�C�
�s33�)C��=                                    Bxs�Q�  �          @�p�@P  �<���33��\)C��=@P  �\)�?\)�p�C���                                    Bxs�`D  "          @�@4z��4z��P�����C���@4z����xQ��CQ�C�0�                                    Bxs�n�  �          @�?�33�p  ?���A���C��f?�33���H?G�A ��C���                                    Bxs�}�  
�          @���@}p���Q���H��G�C�s3@}p���\)����C�{                                    Bxs��6  T          @�G�@l���>{�33�Σ�C���@l�����?\)�
(�C�33                                    Bxs���  
�          @�G�@y���<�Ϳ�z�����C�k�@y�����
=���C�                                    Bxs���  T          @��@w��9���L�����C��
@w��0  �s33�4(�C�Q�                                    Bxs��(  �          @�33@c�
�aG��@  ��
C��=@c�
�L�ͿУ���ffC���                                    Bxs���  "          @�@���AG��\)��  C��
@���1녿����n=qC��)                                    Bxs��t  �          @��H@�\)��{?�@�G�C���@�\)��>�@��C��                                    Bxs��  
�          @�Q�@�(�>B�\?uA2{@(�@�(���\)?z�HA4��C���                                    Bxs���  T          @�33@�(����R?�z�A�=qC�5�@�(��(��?��
Ai�C�+�                                    Bxs�f  �          @���@�=q�h��?z�HA5��C��f@�=q����?8Q�A  C�b�                                    Bxs�  T          @��
@�=q� ��>�z�@S33C�'�@�=q� �׾u�,��C��                                    Bxs��  T          @��R@��
�)��>�z�@UC���@��
�(�þ�����ffC��                                    Bxs�-X  
Z          @���@�{�Q�?@  A
{C�0�@�{�  >8Q�@33C���                                    Bxs�;�  "          @���@����(���\)�L��C�8R@�����\�c�
�%��C���                                    Bxs�J�  "          @��H@��?�\?�=qA��
@ʏ\@��=�Q�?�33A��?��                                    Bxs�YJ  T          @��
@����333����=qC��@����$z῝p��a��C��                                    Bxs�g�  T          @�  @>�R�l�Ϳ����yG�C�e@>�R�P  �{��=qC�C�                                    Bxs�v�  �          @�@�z���Ϳ�\)�U�C���@�z���Ϳ�����HC���                                    Bxs��<  �          @�G�@8��� ���0���=qC�~�@8�ÿ�p��S�
�2  C���                                    Bxs���  T          @���@���=u�33�Ǚ�?L��@���?z���H���@�                                    Bxs���  "          @�  @��>aG��޸R��33@8Q�@��?&ff��\)��p�A��                                    Bxs��.  "          @��@�G�?�R�(���RA�@�G�?�(������{A}G�                                    Bxs���  
Z          @��
@��H?z�H��=q���RAK�@��H?�z�\���A�                                      Bxs��z  "          @�(�@�?�{�!G�����Ab{@�?��R���
��G�Ay��                                    Bxs��   T          @�p�@�?�G���{���A�@�?�\)���R�w�A��                                    Bxs���  T          @�z�@0��?���j�H�K�RA���@0��@��QG��.BQ�                                    Bxs��l  �          @tz�@X��?}p����
��A��R@X��?�(��=p��:�HA�Q�                                    Bxs�	  
�          @�@g
=?�녿����A��@g
=?�׿8Q�� (�A�                                    Bxs��  �          @�=q@g
=@
=���
��=qB�@g
=@&ff�&ff�Q�BG�                                    Bxs�&^  
Z          @�(�@`��@�
��Q����
A�p�@`��@
=�aG��;�Bz�                                    Bxs�5  
�          @�@��?����p���EA�z�@��?�33�(���
=A�ff                                    Bxs�C�  �          @���@P��@{���H��B�@P��@4z῅��Y�B#z�                                    Bxs�RP  
�          @���@�z�W
=������C��q@�z�>��R�z���p�@��H                                    Bxs�`�  �          @���@p  ��H��
��
=C��{@p  ��\�5�	�RC�aH                                    Bxs�o�  T          @�=q@l(��Dz`\)�~�RC�q@l(��(Q����(�C�>�                                    Bxs�~B  "          @��@��\���H���R�f�HC�(�@��\��{�Ǯ����C�5�                                    Bxs�  Q          @�\)@�
=��(�����Q�C�� @�
=��33��
=��
=C�#�                                    Bxs�  W          @���@�  �(�?�A�G�C�(�@�  �p��?�p�Ao�C�
                                    Bxsª4  �          @�{@�33�0��>�G�@�ffC���@�33�G�>��@J�HC�0�                                    Bxs¸�  T          @���@�����>���@e�C���@�����>B�\@  C�.                                    Bxs�ǀ  
�          @���@�Q�>���>��H@��\@hQ�@�Q�>.{?��@љ�@z�                                    Bxs��&  T          @�@�(�?��H?
=@�z�A��@�(�?��\?n{A/33At��                                    Bxs���  
�          @�
=@�\)@��>�\)@C33A£�@�\)@   ?L��A��A��                                    Bxs��r  �          @�{@�\)@!G�?\)@�  A���@�\)@33?�AP��Aم                                    Bxs�  
�          @�  @�=q@#33>���@j=qA�@�=q@��?s33A'�
A�G�                                    Bxs��  �          @���@��@
=>�Q�@��A�=q@��?���?\(�A{A�\)                                    Bxs�d  T          @��@��H@   ?z�HA1�A��R@��H?ٙ�?�Q�A�\)A��R                                    Bxs�.
  
�          @�p�@��H?�p�?�\)A}G�A�
=@��H?���?�Q�A���AW\)                                    Bxs�<�  �          @���@��\?p��@%�A�\)AB�H@��\>�=q@.�RB �\@fff                                    Bxs�KV  T          @���@��?�{@�\A�G�A�Q�@��?��R@*�HA���A�z�                                    Bxs�Y�  �          @���@�{?��
?��HA�ffA�Q�@�{?aG�?�p�A�G�A333                                    Bxs�h�  �          @�@��?Q�?�\@�\)A#33@��?+�?0��Az�A�                                    Bxs�wH  T          @��R@�p�?(��(���z�@ᙚ@�p�?��׿�(����RAN=q                                    BxsÅ�  
Z          @�  @���?�����(�AQ��@���?�\)�޸R��\)A�                                      BxsÔ�  T          @�{@�{@�
��{�h��A�p�@�{@$z�@  ���A噚                                    Bxsã:  �          @���@���?��\)��(�A��H@���?��>�{@j�HA��H                                    Bxsñ�  
�          @�ff@��\?�\)>��R@S�
A�G�@��\?�p�?B�\A ��A���                                    Bxs���  �          @��@���?�
==L��?�A|Q�@���?�\)>Ǯ@���As
=                                    Bxs��,  "          @�\)@�\)?�
=���\�f=qA�ff@�\)?�Q�aG��33A���                                    Bxs���  T          @��
@��\?�
=���R�d(�A��@��\?�
=�J=q���A�G�                                    Bxs��x  T          @�
=@���?���?#�
@��\A`Q�@���?p��?fffA/�A;�                                    Bxs��  �          @��
@���?���@ ��A�  Ag33@���?z�@{A��
@��R                                    Bxs�	�  "          @�(�@��?�?�
=A�ff@�ff@��=��
@ ��A�Q�?��
                                    Bxs�j  
�          @�p�@xQ�?�ff@G
=B�A�Q�@xQ�?�\@U�B!�H@�                                    Bxs�'  T          @���@g�?�\@eB1��@�ff@g��\@g
=B2�C�                                      Bxs�5�  
Z          @�{@�z�?��
>#�
@�A��@�z�?��H>��@�A�{                                    Bxs�D\  !          @��@���?�(��E���
Aw\)@���?�
=�,(����A��                                    Bxs�S  �          @�(�@�  ?��
�Tz��\)A��\@�  @�\�6ff����A��                                    Bxs�a�  T          @�G�@��H@�
��\�îA�G�@��H@#33��(���G�A�\                                    Bxs�pN  �          @��\@���@�ÿǮ��{A��H@���@,�Ϳn{�ffA�                                    Bxs�~�  
�          @��H@�
=?�p��Fff�{Aup�@�
=?�Q��-p���RA��                                    Bxsč�  
�          @���@p��?O\)�}p��7(�AA@p��?޸R�i���%G�A�ff                                    BxsĜ@  
�          @���@]p�?E���
=�H{AI�@]p�?�\�z=q�4�HAأ�                                    BxsĪ�  
�          @���@Z=q?#�
��G��L  A*=q@Z=q?�z���Q��:=qA�
=                                    BxsĹ�  �          @�p�@9��>����G��dp�A�\@9��?�ff��G��RQ�A���                                    Bxs��2  �          @���@E?W
=���H�V\)As�@E?�{��Q��@�A��\                                    Bxs���  
�          @�Q�@X��?n{�y���@{Au�@X��?��c�
�+  A�ff                                    Bxs��~  T          @�{@Z�H?aG���(��E��Af�\@Z�H?��r�\�1G�A���                                    Bxs��$  �          @�\)@&ff?z�H����l�RA�\)@&ff@z���G��P��B33                                    Bxs��  
�          @�Q�@8��?\���
�U  A�@8��@!G��xQ��5=qB$�\                                    Bxs�p  "          @�33@!G�?��������m{A��@!G�@p����H�K��B1ff                                    Bxs�   "          @���@Q�?������p\)A�33@Q�@(������N
=B6�                                    Bxs�.�  "          @��@�Q쿦ff�z����C�]q@�Q쿏\)�^�R���C�>�                                    Bxs�=b  "          @��\@��R������
�L��C��)@��R��ff��\)�0��C���                                    Bxs�L  �          @�G�@���(��>aG�@{C���@���0��=u?
=C��                                    Bxs�Z�  "          @���@�33�Y��?�ffA((�C��@�33��=q?Q�AffC��                                    Bxs�iT  "          @�ff@��E�?��Az�HC��
@���\)?��AT��C�0�                                    Bxs�w�  �          @���@�녿�ff@G�A�G�C��@�녿޸R?�33A�G�C�}q                                    Bxsņ�  �          @�z�@������@,(�A�p�C�\@���(�@
�HA���C��                                    BxsŕF  
�          @�@��\��33@�A���C�� @��\���?�A�=qC��                                    Bxsţ�  
�          @�33@��H��\@(��A��C�p�@��H�ff@��A�z�C�n                                    BxsŲ�  T          @���@tz���H@J=qBC��\@tz��G
=@\)A֏\C�p�                                    Bxs��8  T          @�  @\(��=q@`  BG�C�� @\(��J�H@4z�A��HC��                                    Bxs���  T          @�G�@`�׿�@s�
B/p�C�'�@`���,��@P��Bz�C�9�                                    Bxs�ބ  
�          @�G�@R�\���@hQ�B&{C���@R�\�P  @<(�B��C���                                    Bxs��*  �          @�  @L���@  @S�
B(�C�t{@L���l��@   A�z�C�q�                                    Bxs���  T          @��@:�H�>�R@<��B��C�5�@:�H�fff@	��A�G�C���                                    Bxs�
v  "          @��H@z��33@B�\B+�HC��H@z��=p�@=qB�C�                                    Bxs�  �          @�ff?���{��  �t��C���?����0  ��(�C�3                                    Bxs�'�  
�          @���?(����\��{�Z{C�q?(���33�)���ڏ\C�h�                                    Bxs�6h  �          @�
==�����G������RffC���=��������,(���\)C��\                                    Bxs�E  T          @�\)<����H��
=�7\)C�%<���z��"�\���
C�'�                                    Bxs�S�  
�          @�z�@&ff���@^�RBL33C��
@&ff���@G�B0�HC��R                                    Bxs�bZ  �          @���@XQ��1�@=p�B=qC�E@XQ��Y��@p�A�  C�c�                                    Bxs�q   
�          @���@"�\��@!�A�G�C���@"�\���?�(�Aw�C�H�                                    Bxs��  �          @��R@���z�@�A�
=C��q@�����?���A5C�H                                    BxsƎL  �          @���@E��r�\?�A���C��@E���z�?uA)�C�W
                                    BxsƜ�  �          @�p�@G
=�l(�@p�A؏\C��@G
=��p�?\A�(�C�W
                                    Bxsƫ�  T          @��@����}p��=��C�Ф@�p  ��Q���p�C���                                    Bxsƺ>  �          @��@>�R���R�E�#  C�#�@>�R�����^{�<�HC��                                    Bxs���  �          @�{@��\�녿�����C��H@��\��ff�{��p�C��
                                    Bxs�׊  �          @�33@c�
�Z=q��(����C�"�@c�
�Mp���p��eC���                                    Bxs��0  
�          @��?������þ����33C�E?����������
����C��
                                    Bxs���  �          @���?Ǯ��Q�fff�#
=C��?Ǯ����   ���C���                                    Bxs�|  
�          @��
?�Q���\)�\)�׮C��=?�Q���
=������(�C�l�                                    Bxs�"  �          @���?�(���ff?�z�A�
=C���?�(���\)?(�@��
C��                                    Bxs� �  
�          @�녿O\)�e@J�HB"�C���O\)���@�RA�Q�C��R                                    Bxs�/n  T          @�ff?(����p�@�A�  C�� ?(������?}p�AB�HC�%                                    Bxs�>  �          @���?�=q�1G�?h��A�p�C���?�=q�9��>aG�@�  C�S3                                    Bxs�L�  "          @������e�@!G�A�\Cl=q������\?���A�\)Co�{                                    Bxs�[`  T          @�G��\��z�@�\A��CyǮ�\��=q?��RAg
=C{��                                    Bxs�j  T          @��R�˅���?��A�(�Cy��˅���
?\)@ҏ\Cz�q                                    Bxs�x�  �          @��Ϳ�(�����?��APQ�CuG���(���p�=#�
?�Cu�3                                    BxsǇR  �          @�
=���s�
?!G�A�HCp�����vff�����uCq#�                                    BxsǕ�  
�          @��\��p��~�R��
=��G�C�W
��p��e����\C�                                    BxsǤ�  T          @���@U��  ?W
=A=�C��
@U����>�z�@�Q�C�,�                                    BxsǳD  �          @P��@0��?333>���A Q�Ac�
@0��?
=?\)A2ffA?�                                    Bxs���  "          @]p�@Y��>�>�@�A z�@Y��>�G�>u@z�H@��                                    Bxs�А  
(          @s�
@q�>�33<�>��@�=q@q�>�{=�G�?�z�@�=q                                    Bxs��6  T          @~{@hQ�?��R=#�
?&ffA��R@hQ�?�Q�>���@��A���                                    Bxs���  �          @��H@s�
?�p�=L��?(��A�p�@s�
?�>���@��A��
                                    Bxs���  
Z          @�=q@s�
?��>#�
@��A�  @s�
?��>�@�33A��                                    Bxs�(  �          @��@]p�@
=<#�
>\)B	\)@]p�@�\?�@��\B(�                                    Bxs��  �          @�p�@P��@&ff=�G�?�p�Bp�@P��@   ?8Q�A�B�                                    Bxs�(t  
�          @���@��@^{�.{�(�BiG�@��@Z=q?&ffA�Bg�\                                    Bxs�7  T          @��@"�\@H��=�?�BK�@"�\@AG�?\(�ADz�BG��                                    Bxs�E�  �          @�
=@QG�@;�?=p�A�B'ff@QG�@*�H?�A�(�B�H                                    Bxs�Tf  
(          @�{@n�R@Q�?�=qAa��A��@n�R?�?�ffA�=qA�33                                    Bxs�c  �          @�z�@|��?�Q�?�33A�G�A��@|��?�ff?���A��
A�p�                                    Bxs�q�  
�          @�\)@�
=?��R?У�A���A�p�@�
=?Q�?��A�33A0(�                                    BxsȀX  T          @�=q@p  >��R@#33B�@�ff@p  ���R@#33B�C��=                                    BxsȎ�  T          @���@`�׿&ff@*�HB=qC���@`�׿�G�@(�B��C�)                                    Bxsȝ�  T          @�p�@�ÿ�=q@(��B%33C�L�@�����@Q�B C�l�                                    BxsȬJ  �          @�(�@>{?�G�@9��B �A�G�@>{?��\@N�RB7ffA�                                    BxsȺ�  
�          @�{@!녾aG�@w�Bb�
C��@!녿���@mp�BT��C��R                                    Bxs�ɖ  "          @�\)?�  ��R@��B��=C�'�?�  ��@��RB�\)C���                                    Bxs��<  �          @���@mp�?
=q=�\)?�=qAz�@mp�?�\>L��@AG�@�G�                                    Bxs���  �          @���@��׾\�   ��=qC�Ff@��׾�  �z���\C�33                                    Bxs���  	�          @�33@0���hQ�333�
�HC��q@0���W
=�����  C���                                    Bxs�.  
�          @Å@/\)��Q쿵�W33C��H@/\)�����(Q���Q�C��                                    Bxs��  �          @�Q�@@����(��n{�	p�C���@@����  ����{C��=                                    Bxs�!z  �          @�\)@N{������{�I��C���@N{��녿˅�m�C�5�                                    Bxs�0   �          @��@{���ÿ�33�%C��3@{���H�\)��C�j=                                    Bxs�>�  �          @�
=@������c�
���RC�C�@����������33C��
                                    Bxs�Ml  T          @�z�@#�
����(���
=C��)@#�
����L����C��                                    Bxs�\  �          @�33?xQ���p��   ����C���?xQ���(���33���C��                                     Bxs�j�  �          @\?#�
��{�k��z�C�?#�
������
��  C�E                                    Bxs�y^  �          @�?333���H��z���(�C�n?333��
=�P  ��C���                                    BxsɈ  �          @�33?���=q�p����C�
=?���z��Z=q���C�z�                                    Bxsɖ�  �          @���?!G���  �0����p�C��R?!G���ff�w��)z�C�Ff                                    BxsɥP  �          @�33?(����Vff�C���?(��o\)��(��Dz�C���                                    Bxsɳ�  �          @���?\)�~{�s�
�.\)C��)?\)�<(������e  C�Y�                                    Bxs�  �          @��
?W
=��G��P  ���C�� ?W
=�����p��3ffC�o\                                    Bxs��B  �          @�  >��
��녿��
�=C���>��
���\�,(���p�C��=                                    Bxs���  �          @�{?�=q���\�������C�E?�=q�����^{��C���                                    Bxs��  �          @�  ��Q���z���H��C��콸Q�����Dz����C��f                                    Bxs��4  �          @��H>W
=��\)��H���RC�R>W
=����j=q�C�C�                                    Bxs��  �          @��?�����
�{��  C���?�����
�k��G�C�|)                                    Bxs��  T          @��>��R��������{C��R>��R����n{�p�C��
                                    Bxs�)&  �          @���?z���Q��n�R�\)C���?z��^�R��ff�TQ�C��)                                    Bxs�7�  �          @���=�����33�c�
��C��q=����g
=����N�C�Ǯ                                    Bxs�Fr  �          @�=q>Ǯ�����HQ����C�&f>Ǯ��  �����4G�C���                                    Bxs�U  �          @ə�?������O\)���
C�4{?����p����3(�C��                                     Bxs�c�  �          @ƸR?�(���(��O\)����C��{?�(���ff��(��4(�C�q                                    Bxs�rd  �          @�{?p����
=�8Q���=qC�
?p�����
����(��C��                                    Bxsʁ
  �          @�(������H����ffC��������\�qG��C�h�                                    Bxsʏ�  �          @��Ϳp����\)��Q���  C����p�����\�Tz���{C�H                                    BxsʞV  �          @�G���p����R��\�{�C{�
��p�����I�����Cy�                                    Bxsʬ�  �          @љ���z���ff����
=C|0���z����\�N{���HCzE                                    Bxsʻ�  �          @љ������������
=C|^��������\�j=q�p�Cz�                                    Bxs��H  �          @Ӆ�-p���  ��=q�`  Ct� �-p���{�;���Q�Crp�                                    Bxs���  �          @׮��
�Å�����\Q�CyQ���
��G��A���Q�Cwff                                    Bxs��  �          @�  �Q���p������HCx
�Q����R�c33���CuxR                                    Bxs��:  "          @�������R�\����Cv(����33��\)�(�
Cq��                                    Bxs��  �          @��H�(���
=�~{�=qCu&f�(��g�����F�Cn޸                                    Bxs��  �          @��Ϳ�(���33�?\)��  Cz�׿�(����R��Q��p�CwQ�                                    Bxs�",  �          @�z��
=����HQ���RC}J=��
=����z��%Cz8R                                    Bxs�0�  �          @��
�G���  �   ��z�Cz���G���
=�tz���
Cw�)                                    Bxs�?x  �          @�z��!�����(�����HCu@ �!����y����
Cq��                                    Bxs�N  �          @��Ϳ������'����C|\�����  �}p��
=CyaH                                    Bxs�\�  �          @�{�/\)��
=��z�� ��Cu\)�/\)��Q��%���(�Cs�\                                    Bxs�kj  �          @�p��,(����Ϳ�ff�V�HCu���,(���33�<(���G�CsT{                                    Bxs�z  T          @�p���
��ff��=q��  Cx��
���\�N�R��z�Cv�                                    Bxsˈ�  �          @Ӆ��
=���R�.�R���HC{Q��
=���
��G���Cxc�                                    Bxs˗\  T          @�p���  �ȣ׿�\)�aC�C׿�  ��{�Fff�ޏ\C33                                    Bxs˦  �          @�p������p������\)C
��������U���Q�C}n                                    Bxs˴�  �          @��Ϳ����  �Q����C�{�����\)�qG��
�\C}�{                                    Bxs��N  �          @��Ϳ�z������+���Q�C�C׿�z���=q��G��
=C~h�                                    Bxs���  �          @�ff�\��  � ������CǮ�\��ff�z=q�(�C}�3                                    Bxs���  �          @ָR����z��/\)����C~(�����G���33�C{��                                    Bxs��@  �          @ָR�����ff�1G���p�C���������\��z����C33                                    Bxs���  �          @�p���(�����(�����HC�9���(���������  C�S3                                    Bxs��  �          @���:�H��\)�5��{C��=�:�H���H��
=�z�C��3                                    Bxs�2  �          @׮���H��p��=q���C�o\���H��z��w
=��RC��f                                    Bxs�)�  �          @ָR�8Q������6ff��33C��)�8Q���z�����33C��                                    Bxs�8~  �          @׮��  �����333��ffC�Q쿀  ��p����R�p�C���                                    Bxs�G$  �          @�=q>�녾�ff?&ffB:p�C�B�>�녿��?�B�\C�]q                                    Bxs�U�  �          @��@�@HQ�@L��B!{B_ff@�@\)@w�BM�B:33                                    Bxs�dp  �          @��R@?\)?��
@J�HB)(�A���@?\)?s33@`��B@�A��\                                    Bxs�s  �          @���@1�@1�@<(�B(�B3�@1�?�(�@b�\B8\)B��                                    Bxś�  �          @��@G�?�33@X��B2��A�33@G�?�@hQ�BC�RA{                                    Bxs̐b  �          @��\@>�R@J�H@)��A�Q�B;{@>�R@�H@VffB$p�B{                                    Bxs̟  �          @�{@�\)=��
@=qA�\?�  @�\)��@
=A��HC���                                    Bxs̭�  �          @���@�z�?��R@33A���A{�
@�z�?0��@�\A�Q�A�R                                    Bxs̼T  �          @��H@��\?E�@��A�  A,  @��\>\)@#�
B p�?��R                                    Bxs���  �          @�\)@n{?�@0��B��A�ff@n{?��@HQ�B��A���                                    Bxs�٠  �          @�ff@w
=@�R@S�
B33B�@w
=?�=q@u�B*{A��
                                    Bxs��F  �          @�33?c�
��������C��?c�
��  ���H��z�C�>�                                    Bxs���  �          @�=q�\)��(���
����C�Ϳ\)��{�U�  C���                                    Bxs��  �          @�
=�8Q������
=��z�C�y��8Q����H�_\)��C��)                                    Bxs�8  �          @�(���
=���Ϳ����r�HC��{��
=����A����C���                                    Bxs�"�  �          @�녿   ���׿ٙ�����C��=�   ����E��z�C�@                                     Bxs�1�  �          @�ff>\)��G���{�K\)C��=>\)��  �5����C���                                    Bxs�@*  �          @�?8Q���ff��Q��W�
C�n?8Q������8����p�C��f                                    Bxs�N�  �          @�?��
���\��G���Q�C��q?��
��ff�J�H���HC��{                                    Bxs�]v  �          @��?^�R����������=qC�q�?^�R�����L(���C�                                    Bxs�l  �          @�
=?(�����R���H�m�C�o\?(����p��2�\��z�C��\                                    Bxs�z�  �          @��>�Q���G����H��G�C�+�>�Q���Q��
=���HC�N                                    Bxs͉h  �          @�ff@���Q쿆ff�P(�C��@��g����Σ�C�'�                                    Bxs͘  �          @�G�?�R�����Y����
C��f?�R����\��{C��{                                    Bxsͦ�  �          @�ff�Ǯ��Q��333���C��H�Ǯ�j=q�vff�9(�C���                                    Bxs͵Z  �          @�Q�?�33���R?ǮA{�C�f?�33��>B�\?��C���                                    Bxs��   T          @�@C�
�vff@1G�A���C�8R@C�
��?ٙ�A�=qC�O\                                    Bxs�Ҧ  �          @�=q@|�;�G�@C33B(�C���@|�Ϳ��H@5B
{C��H                                    Bxs��L  �          @���@����Q�@7�B
=C���@����\@$z�A��C�Ff                                    Bxs���  �          @�z�@�Q�޸R?�{A��C�p�@�Q��?�{AJ=qC��=                                    Bxs���  �          @���@�z���?�p�AW�C�l�@�z���H?!G�@�z�C�:�                                    Bxs�>  �          @�z�@�(��/\)?�R@ʏ\C�XR@�(��333���Ϳ�G�C��                                    Bxs��  �          @�{@����*�H>�=q@/\)C�f@����)����
=��
=C�)                                    Bxs�*�  �          @�Q�@�ff��ͼ#�
�L��C�^�@�ff���!G���=qC��                                     Bxs�90  �          @��H@����(��k��  C�� @����33�Y�����C�B�                                    Bxs�G�  �          @�ff@����*�H=��
?W
=C���@����'
=�(��ÅC�H�                                    Bxs�V|  �          @�Q�@��R�0�׾�p��y��C��{@��R�$zῈ���4(�C���                                    Bxs�e"  T          @���@���Fff��
=�E�C�n@���,(���Q����RC�H�                                    Bxs�s�  �          @�  @�(��Z=q��(��L��C�:�@�(��>�R�33���C�
                                    Bxs΂n  T          @�G�@�=q�z=q��Q��>�RC�3@�=q�^{�
=q��  C��f                                    BxsΑ  �          @�
=@���p�׿��>�RC��
@���U��ff��ffC�O\                                    BxsΟ�  �          @��\@^{�|(�����G�C���@^{�XQ��(����\C��                                    Bxsή`  �          @��H?��H�l���g��#\)C���?��H�(Q���ff�V\)C�Q�                                    Bxsν  �          @��@mp��4z�?Tz�A"ffC�aH@mp��<(�=�\)?c�
C��                                    Bxs�ˬ  �          @Ӆ@���^{?��HAo�
C���@���s33?@  @�Q�C�XR                                    Bxs��R  �          @�=q@��H��(���\)�(�C�w
@��H�z=q����=G�C�5�                                    Bxs���  �          @��H@���G���Q��H��C�@��s33��33�D��C��
                                    Bxs���  �          @�33@��R�q녿}p��  C�U�@��R�Y����Q���33C��\                                    Bxs�D  �          @���@����\)�xQ��	p�C�Ф@����e��p����
C�=q                                    Bxs��  �          @�p�@�����\)������C���@�������
=�i��C�                                    Bxs�#�  T          @Ǯ@�������0����33C��@�����������C���                                    Bxs�26  �          @�z�@������\�0����p�C��@��������{���C���                                    Bxs�@�  
�          @���@z=q���Ϳ^�R�=qC�@z=q��Q���
����C�E                                    Bxs�O�  �          @��H@a���������\��C�Y�@a���{�)����Q�C��                                    Bxs�^(  �          @���@J�H�����(��ap�C�l�@J�H��33�-p��ծC�
=                                    Bxs�l�  �          @��@Z�H������  �ffC���@Z�H���
��R���C��                                    Bxs�{t  �          @���@l����p��}p���C�33@l������(���Q�C��                                    Bxsϊ  �          @���@S33�����������C���@S33��Q��AG���p�C��H                                    BxsϘ�  �          @���@J=q��p������{C��@J=q�{��S33��C�]q                                    Bxsϧf  �          @�{@�R��G��\)��{C��
@�R�|���j�H���C��                                    Bxs϶  �          @��
@.{��=q�5���C�R@.{�Z=q�w��&\)C�Ff                                    Bxs�Ĳ  �          @��?k�����j=q���C�
=?k��K����Z�C�
                                    Bxs��X  �          @\��=q�Vff��33�Uz�CxaH��=q��Q����B�Ck��                                    Bxs���  �          @��H�   �n�R�����O{C�,Ϳ   �z���33{C��                                    Bxs��  �          @�G����
�w����
�G��C������
�\)��\)C�e                                    Bxs��J  �          @��H�#�
�x������Hz�C�ٚ�#�
�   ����B�C�5�                                    Bxs��  �          @��
<#�
�dz����R�X��C��<#�
�
=��\)�C�)                                    Bxs��  �          @��?�ff�z�H@�=qB��
C��?�ff�Q�@��Bi�C���                                    Bxs�+<  �          @�\)@C33�AG�@UB��C���@C33�s�
@=qA�(�C�\)                                    Bxs�9�  �          @��
@p  �Q�@=qA�Q�C�aH@p  �s�
?�
=Ap(�C�@                                     Bxs�H�  �          @�{@�Tz�@z=qB/�
C��q@��  @7�A��C�n                                    Bxs�W.  �          @��
@7
=��@z�HB;
=C�h�@7
=�S33@I��B
=C�o\                                    Bxs�e�  �          @���@\(��k�@�{BF�C���@\(��G�@r�\B-��C��                                    Bxs�tz  �          @��R@.{�u@?\)BC���@.{��  ?�A�  C���                                    BxsЃ   T          @��?�Q���p�@
=A�C��H?�Q����?0��@�\)C��                                    BxsБ�  �          @�Q�?�����@L(�B	�RC���?���
=?�33A�{C�%                                    BxsРl  T          @���?����\)?�A��RC�3?�����?��@�p�C�n                                    BxsЯ  T          @�Q�@���=q@L��B
�
C�{@���G�?�(�A��C�:�                                    Bxsн�  �          @�{?�Q���Q�@.{A�  C��?�Q����\?��Adz�C�t{                                    Bxs��^  �          @�Q�?�G���Q�   ��z�C��{?�G�����������C�T{                                    Bxs��  �          @���?�(����Ϳ���/33C��3?�(���p��=q�ѮC���                                    Bxs��  T          @�=q@J�H�����Q�uC��@J�H��p����
�\Q�C���                                    Bxs��P  �          @���@n�R��{>�z�@>�RC��
@n�R����^�R��HC��                                    Bxs��  �          @���@g
=����\)��z�C�z�@g
=�}p����
�V=qC�,�                                    Bxs��  �          @�33@6ff���\���
�O\)C��{@6ff��(�����e��C���                                    Bxs�$B  �          @�(�@333����<�>�\)C�7
@333���H����X(�C��3                                    Bxs�2�  �          @�=q>�
=���@1�A��C�aH>�
=��=q?��AO\)C�#�                                    Bxs�A�  �          @�p�?�G���\)?(��@�33C�O\?�G���ff�O\)��RC�XR                                    Bxs�P4  �          @��?�������?��A\)C��?������<��
>#�
C���                                    Bxs�^�  T          @�\)@ff����@(�A��C��@ff���?��AD��C�Ф                                    Bxs�m�  �          @�@�R����?��A�z�C��@�R��Q�?&ff@���C�%                                    Bxs�|&  �          @��?�(���\)?�G�A/�
C�<)?�(���=q��
=��=qC��                                    Bxsъ�  �          @���?�  ���R�����C�s3?�  ���
�����RC���                                    Bxsљr  �          @�{�+�����c�
���C��{�+��I�����
�\�C���                                    BxsѨ  �          @��ýu���\��  �1��C��R�u�=p���G��s  C�h�                                    BxsѶ�  �          @Å>������xQ��{C��\>���]p������^\)C�G�                                    Bxs��d  �          @�33?�\�����`����C�7
?�\�W������V��C�P�                                    Bxs��
  �          @�
=�#�
�~{�>�R���C��
�#�
�>{�~�R�U(�C���                                    Bxs��  �          @�G�@   �\)@P  B(�C�P�@   ���?��RA��HC�n                                    Bxs��V  �          @��H?�
=��\)@+�A�{C��\?�
=����?�ffAXQ�C�y�                                    Bxs���  �          @�=q?����  @�A��RC���?����p�?=p�@�C��q                                    Bxs��  T          @�(�?�p���{@
=A���C��?�p���?�  A*=qC���                                    Bxs�H  T          @�(�@�`��@7�B	�C���@��?�G�A��C��
                                    Bxs�+�  �          @��R@<(��x��?���A���C���@<(���z�>�z�@Mp�C��\                                    Bxs�:�  �          @�Q�@[��y��?   @�{C���@[��xQ�#�
��G�C��
                                    Bxs�I:  �          @�\)@dz��j�H�!G���G�C��@dz��U���33��C�y�                                    Bxs�W�  �          @��@W
=�3�
�+���C��@W
=���W��$=qC��                                    Bxs�f�  �          @��׾#�
���@���B�{C�p��#�
�,(�@�
=Bk�RC�G�                                    Bxs�u,  �          @��?E��1�@c�
BK�C���?E��l(�@'
=B
�C���                                    Bxs҃�  �          @��H@�X��@8Q�B�\C�N@���\?��
A��C��\                                    BxsҒx  �          @�  @/\)��@fffB:�C��H@/\)�>�R@7
=B�C�O\                                    Bxsҡ  �          @��R@>�R�
�H@i��B3  C���@>�R�HQ�@7�B\)C���                                    Bxsү�  �          @���@@  �'
=@333Bz�C���@@  �S�
?�A�Q�C�                                      BxsҾj  �          @�{@{�HQ�@C33B�
C���@{�xQ�@ ��A���C��)                                    Bxs��  �          @�
=@AG��R�\?��A�
=C�G�@AG��j�H?B�\Ap�C��)                                    Bxs�۶  T          @�
=@p  �:�H?���A��
C��@p  �L��>��@��C�Ǯ                                    Bxs��\  �          @��@J�H�.�R@'
=A�33C���@J�H�W�?ٙ�A���C��                                     Bxs��  �          @�  @>�R�HQ�@�AЏ\C��@>�R�g�?�{AT��C��q                                    Bxs��  �          @���@*�H�/\)?W
=AD��C�!H@*�H�7�    �L��C�~�                                    Bxs�N  �          @��
@p��Tz�\���C���@p��C�
�����=qC��                                    Bxs�$�  �          @��@.{�R�\?!G�A�HC�˅@.{�U���Q�����C��                                     Bxs�3�  �          @��?��H�{��z���=qC�,�?��H�I���HQ��(ffC�l�                                    Bxs�B@  �          @�33@�
�Z�H������z�C�  @�
�7���\��G�C�c�                                    Bxs�P�  �          @�
=@~�R���>��@�(�C��)@~�R��z�=L��?=p�C�9�                                    Bxs�_�  �          @�Q�@���?:�H@!�A�A%p�@��׽L��@(Q�BC���                                    Bxs�n2  �          @�
=@��H?��@ ��A�(�A\(�@��H>�=q@-p�A�33@fff                                    Bxs�|�  �          @�p�@mp�@1�@�A���Bff@mp�@�@6ffB
=A�
=                                    BxsӋ~  �          @�ff@��R@{?�A���Aޏ\@��R?���@=qA��A�{                                    BxsӚ$  �          @��@���ff@G�A��
C�Ff@����
?�ffA�(�C��\                                    BxsӨ�  �          @�ff@��
���\@p�A�p�C�p�@��
�У�?�ffA���C��                                    Bxsӷp  �          @��@�33=�G�?�z�Al��?�33@�33���
?���Ag
=C�J=                                    Bxs��  �          @���@�\)�
=?�\)A7�
C���@�\)�fff?c�
A{C�XR                                    Bxs�Լ  �          @���@�  =#�
<#�
>�>�
=@�  <�<��
>��>�Q�                                    Bxs��b  �          @���@�\)��{>�=q@<��C�*=@�\)����>#�
?��HC��                                    Bxs��  �          @��\@�33��?+�@�C�>�@�33��ff>W
=@  C���                                    Bxs� �  �          @�(�@��Ϳ��R?Tz�AffC�=q@��Ϳ�
=>�
=@��RC�<)                                    Bxs�T  �          @�(�@�=q�@  >8Q�?�33C�H@�=q�E��u�
=C��                                    Bxs��  �          @�\)@�  ���H?(�@���C�8R@�  ��=q>\)?��HC���                                    Bxs�,�  �          @���@�����?(��@�(�C��@�����R=u?z�C�Y�                                    Bxs�;F  �          @�=q@�
=��?:�H@��C�q@�
=����>8Q�?�C�j=                                    Bxs�I�  �          @�G�@��Ϳ�?uAp�C���@����G�>�Q�@g�C��                                     Bxs�X�  �          @�@��׿���?�ffA*ffC�xR@����33>�G�@�\)C�Z�                                    Bxs�g8  �          @�Q�@��\��Q�?s33A��C��)@��\���>��R@HQ�C�                                    Bxs�u�  �          @��\@�(��(��?:�H@�{C�]q@�(��/\)����33C���                                    BxsԄ�  �          @�33@����H��?�@��\C���@����J=q��
=��z�C��3                                    Bxsԓ*  �          @��\@��%?+�@�z�C��3@��*�H�#�
���
C�W
                                    Bxsԡ�  
�          @�33@�ff�   >��H@��C��@�ff�33�����
=C��\                                    Bxs԰v  �          @��R@�p��G���(���
=C�O\@�p��������+�
C�q�                                    BxsԿ  �          @�  @�
=�3�
>W
=@(�C���@�
=�/\)�+���  C��=                                    Bxs���  �          @�  @���1G��.{��  C��
@���%���\�*�HC��{                                    Bxs��h  �          @�z�@�\)�<�;W
=�p�C�O\@�\)�/\)��{�?\)C�C�                                    Bxs��  �          @��\@����Tzᾨ���Mp�C��
@����C33����U�C���                                    Bxs���  �          @���@��R�)�����R�J�HC��=@��R����\)�8  C���                                    Bxs�Z  T          @��@��R�XQ쾣�
�QG�C�p�@��R�G
=��{�`Q�C���                                    Bxs�   �          @���@h�����ÿ����2�\C��@h���_\)����ffC�
                                    Bxs�%�  �          @�p�@n�R��G�����V�RC�b�@n�R�Z�H� �����HC���                                    Bxs�4L  �          @��@l���x�ÿ޸R��p�C��@l���I���7���z�C��3                                    Bxs�B�  �          @��H@�33�N�R��z�����C���@�33�{�5����RC�~�                                    Bxs�Q�  �          @�ff@k��j=q�33��
=C���@k��0  �U����C��                                    Bxs�`>  
�          @�ff@_\)�b�\�3�
��C�N@_\)�{�q��%��C�Y�                                    Bxs�n�  �          @��H@%�\(��b�\��RC�t{@%�	����ff�S�C�&f                                    Bxs�}�  �          @�G�@)���c33�QG����C�W
@)������\)�H��C�L�                                    BxsՌ0  �          @�\)@%�_\)�Tz���C�C�@%�����  �LffC�q�                                    Bxs՚�  �          @�p�@��e��\(��G�C�N@��33�����ZG�C�P�                                    Bxsթ|  T          @�\)@�\�N{�|���8  C�"�@�\��ff�����q\)C�E                                    Bxsո"  T          @�ff@��2�\�����Az�C��@녿�����\)�r��C��3                                    Bxs���  �          @�ff@>�R�&ff���\�[=qC��
@>�R?E�����Y��Af�\                                    Bxs��n  �          @�p�?��Ϳ���ff�q��C��{?��;��
��=qW
C�&f                                    Bxs��  �          @�=q?@  ���
=�y��C��H?@  �333��
=�qC�n                                    Bxs��  �          @��R?�ff�e��~{�;�C�(�?�ff�
=�����C�1�                                    Bxs�`  �          @�33?.{��=q�_\)�33C�xR?.{�>{��ff�d{C�s3                                    Bxs�  �          @��\>��������W
=�  C���>����E���33�`{C��q                                    Bxs��  T          @�z�>����zῷ
=�mp�C�/\>����z��Fff��\C���                                    Bxs�-R  �          @��
?�=q��{�   ���C�� ?�=q��\)��R����C�                                      Bxs�;�  �          @��?!G�����>��@��C�9�?!G���������_\)C�O\                                    Bxs�J�  �          @��>�G���p��Ǯ���C�K�>�G���  �Q���C�}q                                    Bxs�YD  �          @�G���Q���p��:�H���C�� ��Q���������{C�q�                                    Bxs�g�  �          @��ͽ�����33>��R@a�C�t{������p���\)�xQ�C�p�                                    Bxs�v�  �          @�  ?L����{?�33A��\C�g�?L�����\>�(�@�G�C��3                                    Bxsօ6  �          @��>�����>u@<��C��>����zῢ�\��G�C���                                    Bxs֓�  �          @�\)�0���^{� ���Q�C�^��0������`  �W��C~G�                                    Bxs֢�  �          @��>���33�
=q�˅C�� >�����
=��ffC��3                                    Bxsֱ(  �          @��R�#�
����=u?B�\C��#�
�����\���
C��                                     Bxsֿ�  �          @�>�=q��  ?���AP��C���>�=q��33����{C���                                    Bxs��t  �          @��\?G����H�fff�.=qC��=?G���G��
=��33C�o\                                    Bxs��  �          @��H?�R���R�����\)C�*=?�R�P���Z=q�7{C�XR                                    Bxs���  �          @�  ��{���H�@  ��C{����{�g
=�z���p�Cyh�                                    Bxs��f  T          @��\����녿z�H�B�HC�7
���~{�(�����C��                                    Bxs�	  �          @�ff>�33��\)���H�ep�C�
>�33�����.{�  C�o\                                    Bxs��  �          @�(�?.{����!G���C��f?.{����G����C�S3                                    Bxs�&X  T          @���?�\���׿����t  C���?�\�����>�R�
�RC�e                                    Bxs�4�  �          @��u�mp��>{�\)C�"��u�   ��G��hz�C�=q                                    Bxs�C�  �          @�33@<��?G�@i��BH�\An�\@<�;��H@l��BLz�C�N                                    Bxs�RJ  �          @�ff@aG��k�@y��B<{C��3@aG��
�H@W
=B�\C�"�                                    Bxs�`�  �          @�=q?�
=�#�
@��RB��C�c�?�
=���H@���Bt(�C�5�                                    Bxs�o�  �          @���@33��ff@�
=Bs  C�>�@33�4z�@�Q�B>�C���                                    Bxs�~<  �          @�G�?�
=��z�?�(�An�RC�w
?�
=��녾���FffC��                                    Bxs׌�  �          @��?�����\(���RC�B�?�����H��R��C��                                    Bxsכ�  �          @��\?�{�G
=@33B��C�u�?�{�l��?�{Ao\)C�W
                                    Bxsת.  �          @�33@`�׿��R@X��B   C�/\@`���A�@ ��A�33C���                                    Bxs׸�  �          @��
@Dz��XQ�@�A�p�C�%@Dz��y��?W
=AffC�q                                    Bxs��z  T          @��?��
���H    �#�
C�/\?��
���\��p����C��3                                    Bxs��   
�          @�p�?�\)���R�����{�C�y�?�\)�vff��(���C�"�                                    Bxs���  �          @��R=�Q����R��
=��z�C���=�Q���p��B�\�p�C��f                                    Bxs��l  �          @��\>.{��{��R���
C��\>.{������ׅC�
=                                    Bxs�  �          @�p�?\���>��@���C��q?\����(��^�\C��R                                    Bxs��  �          @���@�\��
=?�p�A���C�G�@�\��(�>��
@\(�C�\)                                    Bxs�^  �          @�{@
�H����?}p�A7\)C�"�@
�H��
=�
=��
=C��3                                    Bxs�.  �          @�Q�@�����?�\)AF�RC�@�����R�
=q��C���                                    Bxs�<�  �          @��
@
=���>�  @,��C�s3@
=��ff�����up�C���                                    Bxs�KP  �          @���@<(�����>.{?�p�C�(�@<(���p����H�u�C��{                                    Bxs�Y�  �          @�{@8������?#�
@�G�C�O\@8�����R�xQ��&{C�|)                                    Bxs�h�  �          @��H@����=q?�33AI�C��{@������\��G�C��=                                    Bxs�wB  �          @��\?�(����\?�(�A��C���?�(���z�<#�
=�Q�C���                                    Bxs؅�  �          @�\)?�33���
>�ff@�=qC���?�33��ff��\)�j�\C�=q                                    Bxsؔ�  �          @�(�?������>#�
?�G�C��=?����(���33��=qC��\                                    Bxsأ4  �          @�33?��
����>��R@^�RC��?��
��ff��
=��
=C��                                    Bxsر�  �          @��\@)����
=?��A��
C�{@)������=L��?�C�/\                                    Bxs���  �          @�{@u��s33@P��B\)C��@u���\@.{B Q�C��q                                    Bxs��&  �          @�\)@\(��c33?���A�p�C��@\(��x��>u@*�HC���                                    Bxs���  �          @�(�@L���s33?G�A�C�@L���u��(���\)C��                                    Bxs��r  �          @�@I���}p�?@  A=qC�G�@I���}p��5��p�C�AH                                    Bxs��  T          @���?����p��xQ���
C���?����
=�5����HC�W
                                    Bxs�	�  �          @�=q��Q����H��\���RC�uý�Q��x���XQ��#�RC�T{                                    Bxs�d  �          @�����n{�>�R�33Cs(�����=q��33�UG�Ch�                                    Bxs�'
  �          @�������l(��8���G�Cl������=q�����F��Ca�                                    Bxs�5�  �          @�G����~{�   ���
Cn�=���4z��p���5(�Ce0�                                    Bxs�DV  �          @�=q��\��=q�.�R���\Cv����\�5���Q��H��Cm�                                    Bxs�R�  
�          @�33��  ��Q�� ���ᙚC~xR��  �S�
�}p��@��Cy=q                                    Bxs�a�  �          @�
=�����
�xQ��2�\Cz)���}p��$z�����Cw!H                                    Bxs�pH  �          @��ÿ�(����H�����  C�E��(�����������C~��                                    Bxs�~�  �          @�  ?�����\?W
=A  C�� ?����녿u�/\)C���                                    Bxsٍ�  �          @���?��
��ff=�G�?��C��H?��
���Ϳ�Q���(�C�t{                                    Bxsٜ:  �          @�G�?��H��33��ff��(�C���?��H���H�G���  C�ff                                    Bxs٪�  �          @�
=?�����\�aG���RC�e?�������33���HC��{                                    Bxsٹ�  �          @��=#�
��{>�Q�@�z�C�8R=#�
��\)���H���C�:�                                    Bxs��,  �          @��Ϳu��\)?�@�\)C���u���\�����p  C�h�                                    Bxs���  �          @�(��33��=q���H�O33Cv��33�����8Q����Cs�                                    Bxs��x  �          @����%���R��ff��(�Cnh��%�P���N{�Cg�=                                    Bxs��  �          @�p��z����׿�=q�o
=CuxR�z��n{�8���Cp�q                                    Bxs��  
�          @��?�{���?+�Az�C�@ ?�{��  �u�>ffC�`                                     Bxs�j  �          @�
=?�(�����=p����C��R?�(��e�{���C��                                    Bxs�   �          @���?.{��G�����T��C�>�?.{�tz��+��
��C��                                    Bxs�.�  �          @���?�33�����
=q��p�C�"�?�33�<(��_\)�8
=C��H                                    Bxs�=\  �          @�=q?�
=��=q������C��3?�
=�<���fff�6��C��H                                    Bxs�L  �          @�?��\���R���ffC�xR?��\�Q��tz��@G�C��                                    Bxs�Z�  �          @�{?G���(��&ff���C�\?G��Fff��G��N��C�3                                    Bxs�iN  �          @�?.{�`���o\)�9\)C�xR?.{��{�����C��q                                    Bxs�w�  �          @�{>�Q��Fff�qG��I��C�J=>�Q쿼(������C��{                                    Bxsچ�  �          @��Ϳ0���H���j�H�C33C��Ϳ0�׿�����\u�Cu�f                                    Bxsڕ@  �          @���R�O\)��33��Ch�q��R?xQ���=q�
B�8R                                    Bxsڣ�  �          @�  ?�G�����?�=qA���C�H�?�G������k��*�HC��3                                    Bxsڲ�  �          @�  ?������
>�z�@O\)C�9�?�����(���ff���HC��
                                    Bxs��2  �          @��@=q��ff��������C�<)@=q�~{��\��C���                                    Bxs���  T          @�(�@����N�R����g�
C���@����\)�p��ڣ�C�1�                                    Bxs��~  �          @�p�@�=q�C�
�Y����RC�Q�@�=q� �׿�����Q�C���                                    Bxs��$  �          @�{@�
=�Dzᾏ\)�<��C���@�
=�/\)���rffC�>�                                    Bxs���  �          @�Q�@qG��|�;�����HC�ٚ@qG��`  �����Q�C���                                    Bxs�
p  �          @�
=@�=q�;�>���@�Q�C��=@�=q�7��=p�����C���                                    Bxs�  �          @�
=@��
���?�@��HC�)@��
��(��\)����C���                                    Bxs�'�  �          @��R@n�R�O\)?��RA��C���@n�R�c�
>\)?���C�*=                                    Bxs�6b  �          @��R@~{�"�\@*=qA뙚C��@~{�W�?��RA�  C��)                                    Bxs�E  �          @��H@�����@,(�A�\C�  @����?\)?�
=A��C���                                    Bxs�S�  �          @�Q�@_\)�j�H?��RA��C�Ǯ@_\)����>�G�@�33C�                                    Bxs�bT  �          @�Q�@(Q����@�
A�  C�o\@(Q���z�?z�@��C�                                    Bxs�p�  �          @�=q@5�r�\@<(�A���C�n@5��(�?�G�AQ�C��                                    Bxs��  �          @���@.{�QG�@\(�B�C���@.{��33?�A�Q�C��                                    BxsێF  �          @��@:�H��@�
=B[z�C�{@:�H�333@j�HB)  C�R                                    Bxsۜ�  �          @�Q�@2�\�aG�@�
=Bm�C���@2�\��@���BLz�C��q                                    Bxs۫�  �          @��R@P�׿���@�33BG�C���@P���&ff@VffB
=C���                                    Bxsۺ8  �          @�  @^�R�@c�
B!G�C��@^�R�a�@�A̸RC�G�                                    Bxs���  �          @���@e���z�@mp�B)�
C��R@e��L(�@+�A陚C�)                                    Bxs�ׄ  �          @�\)?�\)?��@��B�ffA��
?�\)��
=@��B�HC�E                                    Bxs��*  �          @�G�?�Q�>�33@�p�B��A9�?�Q��Q�@���B��C�z�                                    Bxs���  �          @��@�H��33@�ffB~�HC��H@�H�  @�p�BT�HC���                                    Bxs�v  �          @��@6ff?z�@�(�Bg�\A9@6ff���\@��B]  C���                                    Bxs�  �          @��@@��>�ff@��RBd�RA��@@�׿�
=@�Q�BVp�C�N                                    Bxs� �  �          @��@Dzᾣ�
@�33B`�C��)@Dz��ff@�33B?G�C��\                                    Bxs�/h  �          @��@	������@�\)B�u�C��R@	���<(�@��BC
=C�{                                    Bxs�>  �          @��
@Dz῀  @��BZ�RC���@Dz��-p�@s33B+ffC�J=                                    Bxs�L�  �          @���@7��u@��
Bc
=C�Ǯ@7��,��@w�B1�C�c�                                    Bxs�[Z  �          @���@1녿5@��RBj��C���@1�� ��@���B<��C��R                                    Bxs�j   �          @�  @=p��&ff@��Baz�C��=@=p����@{�B7z�C�}q                                    Bxs�x�  �          @�p�@Dz�^�R@��BW=qC��@Dz��!G�@j=qB*��C�G�                                    Bxs܇L  �          @���@   ���R@��Ba{C��@   �E@Z=qB"z�C�}q                                    Bxsܕ�  �          @��@�
=�u@>{B33C���@�
=���R@,��A�G�C�E                                    Bxsܤ�  �          @��R@�\)�G�@9��B�
C��H@�\)��\)@ffA�33C��                                    Bxsܳ>  �          @�(�@�
=�333?E�A  C���@�
=�s33>�(�@��HC��3                                    Bxs���  �          @��@�{��@=p�B33C��@�{��R@	��A���C��{                                    Bxs�Њ  �          @��@�
=���k��:�HC�` @�
=���ÿp���<��C��\                                    Bxs��0  �          @���@�ff���
���ǮC�  @�ff��녿#�
��
=C��=                                    Bxs���  
�          @�p�@����33�'���ffC���@���0���J=q�Q�C�S3                                    Bxs��|  �          @�Q�@tz��\�x���+  C��
@tz�u�����@C���                                    Bxs�"  �          @�=q@��
�z��L(��p�C�P�@��
��R�p  �!ffC��q                                    Bxs��  �          @��@����
�:=q��C��)@�녿5�_\)��\C���                                    Bxs�(n  �          @�{@����G�����U�C���@����=q��z����C�(�                                    Bxs�7  �          @�=q@���޸R�O\)� ��C�7
@����G������fffC���                                    Bxs�E�  �          @��@�  �ٙ��Q��{C�k�@�  ��(���Q��eG�C��                                     Bxs�T`  �          @��R@�=q����ff�"�HC��q@�=q��p��ٙ���z�C��f                                    Bxs�c  �          @�p�@��
��\�����C�=q@��
��zῙ���;�C���                                    Bxs�q�  �          @�\)@��H��
=��33�9�C�J=@��H�����p����HC�]q                                    Bxs݀R  �          @�@�  ���#�
��
=C�H@�  ���fff��RC���                                    Bxsݎ�  �          @�z�@�p��(�<��
>k�C��q@�p�� �׿aG����C��                                    Bxsݝ�  T          @��@�����?�{AC\)C��f@���=q>\)?��RC�T{                                    BxsݬD  �          @�z�@�=q�33?�\)A�{C���@�=q�0��?�@�C�o\                                    Bxsݺ�  �          @�
=@���C33?�p�AQ�C�Z�@���R�\����{C�U�                                    Bxs�ɐ  �          @��@��H�Vff?��RARffC�S3@��H�dzᾀ  �)��C�u�                                    Bxs��6  �          @���@��\�!G�?O\)A�HC�:�@��\�(Q쾙���HQ�C���                                    Bxs���  �          @�Q�@�{�L��?���A��RC�K�@�{�dz�=�G�?�33C�˅                                    Bxs���  �          @���@s33�r�\?�p�A��RC���@s33��p�<#�
=���C�1�                                    Bxs�(  �          @�Q�@�\)��@fffB�HC�%@�\)��{@N{B33C�!H                                    Bxs��  T          @�(�@������@s33B+G�C���@�����@Mp�B(�C�33                                    Bxs�!t  �          @��R@u��z�@~{B0C��@u�;�@A�B �\C�U�                                    Bxs�0  �          @��@H�þ�@���B[��C���@H���33@z�HB4�HC��H                                    Bxs�>�  �          @�G�@*�H��
=@�
=Bc\)C��@*�H�;�@dz�B(
=C�'�                                    Bxs�Mf  �          @�\)@���z�@�\)Bq�C���@��^�R@fffB%��C���                                    Bxs�\  �          @���?�z�u@�(�B��C��)?�z��AG�@�Q�BL=qC�q�                                    Bxs�j�  �          @�=q?��;�@�33B�{C�E?����*�H@��BeG�C�ff                                    Bxs�yX  �          @�
=?��ÿ:�H@�{B��3C��
?����<(�@�(�BY33C��                                    Bxsއ�  �          @�p�?�Q�xQ�@�z�B�B�C�?�Q��H��@��BQ�C�S3                                    Bxsޖ�  �          @���?��W
=@�(�B���C���?��:�H@���BO��C�޸                                    BxsޥJ  �          @�Q�?�{��@��
B�  C�3?�{�$z�@�ffBZ(�C��                                    Bxs޳�  �          @�33?У׿^�R@��B�C���?У��@  @�(�BPG�C�L�                                    Bxs�  T          @��@	����z�@�ffBu
=C�*=@	���e@r�\B(�\C�q�                                    Bxs��<  �          @��R@��z�H@��\B|��C��R@��B�\@�{B=�C���                                    Bxs���  �          @�G�@	���z�@��B�z�C�o\@	���*�H@��\BN��C�o\                                    Bxs��  �          @���@
�H�Y��@�  B�\C�XR@
�H�8��@�p�BDz�C�w
                                    Bxs��.  �          @�z�@�ͿW
=@��HB�\)C���@���;�@�  BE(�C�xR                                    Bxs��  �          @��R?}p����@��\B�Q�C�?}p��(��@���BkG�C�E                                    Bxs�z  �          @�Q�?Ǯ��=q@�  B�p�C��?Ǯ��R@�z�Bf�HC�
                                    Bxs�)   �          @���@녽u@��\B�\)C�%@��{@�=qBbz�C�<)                                    Bxs�7�  �          @�p�?�p�>�G�@��
B���A�Q�?�p���33@�G�B�C��
                                    Bxs�Fl  �          @�=q?+���p�@��RB���C�o\?+��1�@�  Bp�C�˅                                    Bxs�U  �          @���@ff��Q�@��B��C�H�@ff�S�
@�
=B<p�C�0�                                    Bxs�c�  �          @�@%��z�@�\)Bw�C���@%��*=q@�  BC�\C��                                    Bxs�r^  �          @��
@�>�33@�G�B��\A��@�����@�ffBmz�C�1�                                    Bxs߁  �          @�  @h���&ff@|��B%��C�=q@h������@��A�C���                                    Bxsߏ�  �          @�
=@]p��&ff@��\B-Q�C���@]p����@#�
A�(�C��                                    BxsߞP  �          @�ff@u�33@\(�B(�C���@u�c�
@
=A��C���                                    Bxs߬�  �          @�p�@]p��'�@hQ�B�C�l�@]p��{�@��A�  C��=                                    Bxs߻�  �          @��\@Z=q�N�R@@  B(�C�E@Z=q���R?�ffAX��C��                                     Bxs��B  �          @�ff@i���J=q@>�RA���C���@i����z�?��AU��C��{                                    Bxs���  T          @��@s33�aG�@'
=A�=qC���@s33����?Tz�A\)C���                                    Bxs��  �          @�\)@z�H�K�@*=qA�(�C�s3@z�H����?�G�A#33C��                                    Bxs��4  �          @�\)@c33�\)@u�B)��C��)@c33�k�@{AΣ�C��{                                    Bxs��  �          @�p�@�
���@��\Bt��C�33@�
�R�\@qG�B,C���                                    Bxs��  �          @��R@$z���@��HBNz�C�9�@$z��|(�@:=qA��C���                                    Bxs�"&  �          @��@o\)�`��@+�A���C�c�@o\)���\?aG�A�C�e                                    Bxs�0�  �          @�Q�@l�����?�ffA{�C�(�@l�����\�Ǯ�x��C�AH                                    Bxs�?r  �          @�33@J�H��\)>L��@33C��@J�H���������C�Ф                                    Bxs�N  �          @�Q�@@�����?�\@�33C��f@@����Q�\���C��)                                    Bxs�\�  �          @��H@�=q�^�R?���A9�C���@�=q�g
=�
=q��C�4{                                    Bxs�kd  �          @�{@,(���p���=q�<��C��\@,(��_\)�:=q��
C���                                    Bxs�z
  �          @�
=?�  �������ՅC��3?�  �5��z�H�K{C��q                                    Bxs���  �          @�=q>#�
�����ff�|(�C���>#�
��\)����°B�C��)                                    Bxs��V  
�          @���=�G��G���
=��C��H=�G�>�=q���
­\)B�p�                                    Bxs��  �          @��?�  ���xQ��!�C�W
?�  ��
=�J�H�
�HC�G�                                    Bxsഢ  �          @�z�?��R����=��
?Y��C�W
?��R��Q������G�C�U�                                    Bxs��H  �          @�=q@J=q��ff��G����C���@J=q�s33�����C��H                                    Bxs���  �          @���@�  �_\)>��?\C�Q�@�  �L(����o33C���                                    Bxs���  �          @�ff@�\)�S�
?0��@�\C���@�\)�Q녿\(����C��                                    Bxs��:  �          @��@����;�?�z�AAC��@����H�þ�=q�333C��                                    Bxs���  �          @�G�@�ff�]p�?Y��A=qC�G�@�ff�^{�G��=qC�5�                                    Bxs��  �          @���@{��r�\?�R@�Q�C�
=@{��j�H�����>�RC��H                                    Bxs�,  �          @�  @����c33?�{A;\)C�J=@����j�H�
=��
=C��                                    Bxs�)�  �          @�Q�@~{�U?��A�  C��R@~{�r�\=���?��C�/\                                    Bxs�8x  �          @���@�Q��:=q@p�A�33C��
@�Q��e?+�@ڏ\C���                                    Bxs�G  �          @���@�=q��\?�(�A�
=C�H�@�=q�4z�?�\@��RC��{                                    Bxs�U�  
�          @���@�\)�(�@1�A�G�C�H@�\)�[�?��Adz�C�z�                                    Bxs�dj  �          @�{@`  �c33@G�A��C�L�@`  ��{>�(�@�  C��                                    Bxs�s  �          @��H@W����R?fffAz�C�W
@W��������8  C�|)                                    Bxsၶ  �          @�Q�@1G���(�?��A�33C���@1G����þB�\��
C��q                                    Bxs�\  �          @����{��ff?��HA�Q�Co��{���׾\��{CqO\                                    Bxs�  �          @�{?@  ��  @:�HA�{C���?@  ���>�ff@�(�C��q                                    Bxs᭨  �          @�G�?�����G�@%�A��
C���?�����p�<��
>.{C��)                                    Bxs�N  �          @��H?W
=��@"�\A���C��)?W
=���׽�Q�W
=C�%                                    Bxs���  �          @��H?z����\@,��A�=qC�n?z�����>W
=@
�HC�                                      Bxs�ٚ  �          @�@Vff�QG�@   A��\C��{@Vff����?=p�@�
=C�˅                                    Bxs��@  �          @�=q@�
=���?.{@�Q�C��f@�
=��33�W
=��C���                                    Bxs���  �          @�33@��Ϳ��?�@�=qC��@��Ϳ�
=����ǮC��3                                    Bxs��  �          @�33@��
��=q?@  @�C��@��
�\=u?
=C��                                    Bxs�2  
�          @�G�@�33��  ?}p�A$��C��
@�33���>�
=@�z�C��R                                    Bxs�"�  �          @���@�G���ff?�Q�AG
=C�g�@�G���p�?z�@�Q�C�5�                                    Bxs�1~  �          @���@�G���33?�  AQ�C��=@�G����>u@�RC���                                    Bxs�@$  �          @��H@�33�dz�>��@$z�C�R@�33�R�\��z��_\)C�5�                                    Bxs�N�  
�          @��@���c�
@J=qBz�C�AH@���\)@��AЏ\C�h�                                    Bxs�]p  �          @��@w
=��@@��Bp�C��
@w
=�Z=q?�\)A��
C�G�                                    Bxs�l  �          @�=q@o\)�^{@ffA��RC���@o\)����>��@-p�C�Z�                                    Bxs�z�  �          @��@n�R�z�H@z�A�33C�˅@n�R��<�>��C��                                    Bxs�b  �          @���@��\�}p�?�G�A!p�C���@��\�~�R�n{�z�C���                                    Bxs�  �          @��
@?\)��
=@\)A�33C��@?\)��  ���
�@  C�s3                                    Bxs⦮  �          @�@Dz���?�
=A�=qC�XR@Dz����R�\)���C��                                    Bxs�T  T          @�G�@<������?��AVffC�Ф@<��������
�$z�C��q                                    Bxs���  �          @���@K����R?���A4��C�f@K���ff��
=�;�
C�                                    Bxs�Ҡ  �          @���@p  �z=q�����7\)C��=@p  �<���2�\��Q�C���                                    Bxs��F  �          @��@��R�|(�=#�
>�
=C�s3@��R�`  �����
=C��                                    Bxs���  �          @�(�@���p  ?aG�A
�RC���@���n�R�u��C���                                    Bxs���  �          @�z�@l������>��
@G
=C���@l�����Ϳ�=q���C��                                     Bxs�8  �          @��H@z�H��Q�?z�@�{C�N@z�H���ÿ�p��k�C�3                                    Bxs��  �          @��@�(��_\)?�=qA/
=C���@�(��e�+��أ�C�XR                                    Bxs�*�  �          @�Q�@������?���An�\C��@�����>���@XQ�C���                                    Bxs�9*  �          @��@��\�,��>\@p  C��@��\�#�
�n{���C��                                     Bxs�G�  �          @�=q@��=q�:�H��Q�C��f@������\��\)C�z�                                    Bxs�Vv  �          @�{@L(����
?�z�A�Q�C�Q�@L(���
=������C�W
                                    Bxs�e  �          @�{@HQ����
?�A6�\C�aH@HQ���33��G��Dz�C�o\                                    Bxs�s�  �          @�ff@:�H��z�>�
=@��C���@:�H��\)�33��C���                                    Bxs�h  �          @�{@����׽�\)�:�HC�R@����H�*�H�י�C�b�                                    Bxs�  �          @�\)@���qG�?h��A��C���@���p  �z�H�  C��                                    Bxs㟴  �          @Å@XQ���=q>.{?ǮC��R@XQ���G��  ��
=C�O\                                    Bxs�Z  �          @�33@qG����?
=q@�33C�C�@qG�����G�����C�5�                                    Bxs�   �          @�z�@^�R����>.{?���C�W
@^�R��Q���R��33C��{                                    Bxs�˦  �          @�z�@tz���\)?O\)@��
C�z�@tz������\�f{C�                                    Bxs��L  �          @Å@����
=?!G�@���C�K�@����\)�����n{C�3                                    Bxs���  �          @�
=@����  ?5@�(�C�@�����\����W�
C��)                                    Bxs���  �          @��@i������?p��A33C�*=@i����\)��(��D��C�ff                                    Bxs�>  �          @�33@c�
��Q�?
=q@���C��@c�
�r�\��
=�u�C���                                    Bxs��  �          @�{@o\)�|��@B�\A�C��
@o\)��p�?J=q@��C��q                                    Bxs�#�  
�          @�G�@j�H���@'�A��
C��@j�H���\>k�@
=C��                                    Bxs�20  �          @Ǯ@p������@�
A��C�C�@p����
=��\)�0��C���                                    Bxs�@�  �          @Ǯ@j�H���@Q�A�=qC�o\@j�H���׾����1�C��                                    Bxs�O|  �          @�
=@L(����@
�HA�  C��H@L(�������
=�vffC�~�                                    Bxs�^"  �          @��@q���z�?�G�A=p�C��R@q�����\)�(��C�}q                                    Bxs�l�  �          @��
@tz���p�?n{AQ�C��@tz���G����W\)C�f                                    Bxs�{n  �          @�{@�
=�����
���\C�Y�@�
=�����R��  C�+�                                    Bxs�  T          @�(�@�Q�У��.�R��
=C�@�Q���K���\C�:�                                    Bxs䘺  �          @�
=@j=q�j=q��{��ffC�|)@j=q�(��J�H�\)C�+�                                    Bxs�`  �          @��H@��,(���p���G�C�c�@���33�A��Q�C�9�                                    Bxs�  �          @�  @�  �޸R>�ff@��
C�aH@�  ��  ��ff���C�`                                     Bxs�Ĭ  �          @�\)@��\��\)��(���(�C��@��\�n{�(��ȸRC�>�                                    Bxs��R  T          @�z�@B�\�^�R�33���
C��
@B�\�33�^�R�-�
C��q                                    Bxs���  T          @���@
�H�J�H��33�ң�C�:�@
�H�����L(��@��C���                                    Bxs��  �          @�=q@�������
=��p�C�J=@���n�R�"�\��33C���                                    Bxs��D  �          @�=q@3�
�y����=q��z�C��@3�
�!��_\)�*C�f                                    Bxs��  �          @�
=@E���\��=q��
=C���@E�,(��e��$�C���                                    Bxs��  �          @��H@��
�
=����5p�C�q�@��
�Ǯ��
����C���                                    Bxs�+6  �          @���@�����ff���R���C�9�@����Y�����ɅC���                                    Bxs�9�  �          @�ff@�z��ff��p���
=C�:�@�z�!G��33��ffC�T{                                    Bxs�H�  �          @�  @����p���R����C��
@�����H����j�RC�e                                    Bxs�W(  �          @�G�@����p���p��s33C���@����ff���
�V�\C��                                    Bxs�e�  �          @��\@�Q�޸R�����p�C��q@�Q쿠  ��=q�\��C�S3                                    Bxs�tt  �          @�p�@�=q��{�����C�\)@�=q��\)��\)�_\)C���                                    Bxs�  �          @�33@�z���;�\)�5�C�o\@�z���
����Y�C�q�                                    Bxs��  �          @��@�z�������@��C��f@�z��33��33�e��C��                                    Bxs�f  T          @��
@��H�8��>B�\?���C�0�@��H�'
=��  �EG�C�e                                    Bxs�  �          @�G�@�G����?��\A�\C��q@�G���ü�����C�n                                    Bxs彲  �          @�p�@�G����H?�=qA��C��)@�G�����?���AMp�C��R                                    Bxs��X  �          @�Q�@�(�����@ffA��C��@�(�� ��?��
AH(�C��H                                    Bxs���  �          @�G�@��=L��@�A��
?�@����  ?�\)A�Q�C��q                                    Bxs��  �          @��@�=q��Q�?:�H@�z�C��f@�=q�33���
�VffC�                                      Bxs��J  �          @�@����e?Q�A Q�C��R@����a녿���&�\C�R                                    Bxs��  �          @�33@�녿�z�@XQ�B��C�s3@���+�@�A�ffC�3                                    Bxs��  �          @�{@\��>�Q�@�
=BV�@���@\��� ��@���B<{C���                                    Bxs�$<  �          @���@��
?��@z=qB(ffA�=q@��
�B�\@�=qB0�\C���                                    Bxs�2�  �          @�33@�33?u@���B.�AS�@�33��33@~{B,
=C�.                                    Bxs�A�  �          @�\)@w
=�1G�?�
=A��HC�*=@w
=�W�>��R@\(�C�xR                                    Bxs�P.  �          @��@3�
��z�У�����C��3@3�
�\���u�#{C��q                                    Bxs�^�  �          @�33@Vff��=q�333��(�C�)@Vff�����P  ����C�                                    Bxs�mz  �          @��H@W
=��(�������C�h�@W
=����L(���ffC��                                    Bxs�|   �          @��
@����  ?�ffA.�\C�~�@�����\�z�H��HC�5�                                    Bxs��  �          @�\)@��\��=q@A�G�C�H�@��\��Q���qG�C��q                                    Bxs�l  �          @�(�@�  �~�R@?\)A�=qC�H@�  ��ff?�R@��C��                                    Bxs�  �          @ָR@��\���
@
=qA��\C�,�@��\���
�����%�C���                                    Bxs涸  �          @��@`��?�\)@�BC�A��
@`�׿L��@���BP�HC���                                    Bxs��^  �          @�z�@�����@�33B��C�0�@�����
@(��A���C���                                    Bxs��  �          @�(�@�\)���@��B  C�p�@�\)��ff@Q�A�z�C���                                    Bxs��  T          @���@����!�@~�RB�C�s3@�����@(�A�(�C�"�                                    Bxs��P  �          @��@�\)��H@{�B(�C�R@�\)���@��A�
=C��)                                    Bxs���  �          @�z�@�Q��=q@uBG�C�+�@�Q���Q�@�A�Q�C��R                                    Bxs��  �          @�@����;�@���B	�\C�1�@������@33A��\C�:�                                    Bxs�B  T          @ᙚ@���   @�(�B  C��\@�����@z�A�ffC�(�                                    Bxs�+�  �          @�
=@��H�Q�@�33B��C�w
@��H���
@
=A��\C���                                    Bxs�:�  �          @��H@�(��@w�B	�HC��)@�(��}p�@
�HA�ffC�%                                    Bxs�I4  �          @�@�z��>{@c33A��C��)@�z���33?�=qAS\)C�޸                                    Bxs�W�  �          @�@��K�@Tz�A�
=C�5�@���p�?��\A)G�C���                                    Bxs�f�  �          @ָR@�  �Vff@=p�Aљ�C�R@�  ��(�?aG�@��C�aH                                    Bxs�u&  �          @��@����g�@*=qA��\C���@�����
=>�@�33C���                                    Bxs��  T          @�\)@�z���Q�@A���C��=@�z������{�@  C�h�                                    Bxs�r  
�          @��@��R�{@��A�Q�C���@��R�U?Tz�@��HC�B�                                    Bxs�  T          @У�@�(��y��@�A��C��\@�(���=q=��
?8Q�C���                                    Bxs篾  �          @�z�@��
�HQ�@A��C�C�@��
�p��>k�@�
C��
                                    Bxs�d  �          @�Q�@�(���
=@ffA�C���@�(��!G�?xQ�A�RC���                                    Bxs��
  �          @�
=@�(��QG�?�ffA��\C�{@�(��n�R���ͿfffC�H�                                    Bxs�۰  T          @�33@�ff����?�Q�AQ�C�3@�ff����k����C��)                                    Bxs��V  �          @�z�@��\�c33?�\A��C��
@��\�|�;�=q�\)C�U�                                    Bxs���  �          @�ff@��\�*=q?�  A}C�B�@��\�K�>.{?�  C�'�                                    Bxs��  �          @�z�@�  �P��?�\)AN�HC�w
@�  �`  �����z�C��                                     Bxs�H  �          @��@���H��?���AiG�C��
@���\�;�{�Z�HC���                                    Bxs�$�  �          @�  @���W�?�A<��C���@���`  �333��Q�C��                                    Bxs�3�  �          @�=q@��H�6ff?�ffAnffC�^�@��H�P  ��G���ffC��q                                    Bxs�B:  �          @�ff@�G���@  A�ffC�Z�@�G��Fff?G�@�
=C��                                    Bxs�P�  �          @�Q�@�����?@  @�p�C��f@����{�
=��ffC��
                                    Bxs�_�  �          @ȣ�@�{�#�
�u�{C�޸@�{����G��`Q�C��                                    Bxs�n,  �          @��@�(�@����R�N�RA�@�(�@  ?xQ�A$Q�A��
                                    Bxs�|�  �          @�(�@��@��=�G�?�{A�\)@��@   ?��A]A�G�                                    Bxs�x  T          @��?��@I�����
���RB�  ?��@7�?�=qA��Bwp�                                    Bxs�  �          @��?��@���>L��@(��B�aH?��@e@�BG�B�8R                                    Bxs��  �          @�p�@~�R@s�
@+�A�B.��@~�R?�33@�  B/Ȁ\                                    Bxs�j  �          @�?��@�\)?�ffAW�B�.?��@u�@r�\B*ffB�.                                    Bxs��  �          @Å@l(�@��\@'
=A��B?Q�@l(�@
�H@��\B5=qA�\)                                    Bxs�Զ  �          @��
@�33@h��@��A��B��@�33?�(�@s33B(�A¸R                                    Bxs��\  �          @�(�@��H@i��?�A}�B{@��H@�@S33B33AӮ                                    Bxs��  �          @�ff@���@E@HQ�A��B�@���?�ff@���B0��A\��                                    Bxs� �  �          @�{@��R>�Q�@5A��@��@��R��33@'�A�33C�(�                                    Bxs�N  �          @��H@�(���=q@UB�C�b�@�(��6ff@�A���C��R                                    Bxs��  �          @�\)@����
=@<(�A��\C���@����\)@33A�G�C��R                                    Bxs�,�  �          @�z�@��H�#�
@(�A�p�C�:�@��H���@ ��A�ffC�P�                                    Bxs�;@  
�          @��H@��?fff@333A�p�Ap�@���+�@7
=A֣�C��q                                    Bxs�I�  �          @У�@��
@ ��@:=qA�p�A��@��
>aG�@a�B(�@Q�                                    Bxs�X�  �          @���@��@N{@	��A���B33@��?�z�@`  B33A�\)                                    Bxs�g2  �          @���@�p�@c�
@!G�A���B�@�p�?�G�@\)B (�A�p�                                    Bxs�u�  �          @�Q�?��@���@�A�B��\?��@x��@�Q�BG(�B���                                    Bxs�~  T          @�?ٙ�@��@8��A��B�.?ٙ�@8��@��Bc
=Bn                                      Bxs�$  �          @�=q@J�H@��@HQ�A�z�BO��@J�H?�{@���BQ��A�33                                    Bxs��  �          @�G�@z�H@~{@�\A���B5�@z�H@{@�Q�B&�HA�{                                    Bxs�p  �          @�Q�@�{@e�@A�(�B"33@�{?�{@w
=B Q�A�p�                                    Bxs�  �          @��R@�z�?���@\)A���A��@�z�>���@7
=A�  @�ff                                    Bxs�ͼ  �          @�z�@��>�(�@��A�p�@�z�@�녿fff@�
A��RC�h�                                    Bxs��b  �          @��H@����  ?\AiG�C��@����.{>\)?�ffC���                                    Bxs��  �          @�녿�33�����O\)�G�C}����33�=q����x�Coٚ                                    Bxs���  
�          @�{���
��33�@  ��\Cz����
�3�
���R�e�HCm��                                    Bxs�T  �          @�Q쿷
=����N�R���HC~�\��
=�/\)���q��Cr��                                    Bxs��  �          @��
����Q������
Cy�ÿ��W����H�G�CpW
                                    Bxs�%�  �          @�
=�	����=q�#�
�ǅCw��	���@����=q�R{Cjz�                                    Bxs�4F  �          @љ����H����j=q�C{�f���H� ������yz�Ck�R                                    Bxs�B�  �          @ə�����������"\)Cs�����\���  CX�                                    Bxs�Q�  �          @���
=q�l(��Q��
=Co�
�
=q������
=�t�CU�                                     Bxs�`8  T          @��H��
=�����Z=q�
(�Cz.��
=�p����\�z�Ch�                                     Bxs�n�  �          @�{�����
=�E���\)Cs{��������H�`p�Cah�                                    Bxs�}�  �          @�G�@�����Ϳ�=q��
=C�{@���W
=�������C���                                    Bxs�*  �          @���@�33�0�׿�{�@z�C��=@�33=��\�W\)?�{                                    Bxs��  �          @Ϯ@�33�G����\���C�z�@�33�\)���
�5�C�`                                     Bxs�v  �          @��@�33��녿0����z�C�'�@�33<#�
�L�����=u                                    Bxs�  
�          @ȣ�@�
=�!G���=q���C�
@�
=���Ϳ\)��(�C�&f                                    Bxs���  �          @�\)@�z�Y��>8Q�?�Q�C�
=@�z�Q녾�z��+�C�,�                                    Bxs��h  �          @�Q�@Å�5?fffA�C��=@Å��=q>Ǯ@h��C���                                    Bxs��  �          @�33@�33���\?�{A!�C�q@�33��z�>�\)@"�\C�aH                                    Bxs��  �          @�33@���>�ff@��C�!H@����R�Tz���\C���                                    Bxs�Z  �          @�  @����O\)?   @�(�C�7
@����A녿��\�<(�C�{                                    Bxs�   �          @��@��\�w
=��G���{C�  @��\�N�R�
=��
=C��q                                    Bxs��  
�          @�z�@w��}p��5��\C�,�@w��@  �+���p�C��                                    Bxs�-L  �          @�{@z=q�J�H����f=qC�t{@z=q���0��� ffC�Ff                                    Bxs�;�  
'          @�(�@�
=��z�?�A��HC�S3@�
=��  ?�p�A�33C��                                    Bxs�J�  �          @�@�{�+�?��AmG�C�W
@�{���?Tz�AG�C�                                      Bxs�Y>  T          @�  @�7
=�����C�3@��p��C�
�H�C�ff                                    Bxs�g�  T          @��H�����z=q�Tz����Cx
���ÿ˅��(��3CaQ�                                    Bxs�v�  T          @����������#�
���HC�����\��p��o�RCs�                                     Bxs�0  "          @�ff?u���
�1��p�C���?u��\�����}��C���                                    Bxs��  �          @�(���������=q�.��C|k�����������Hu�Ca�=                                    Bxs�|  �          @�G�?�������Dz�����C��3?���7���(��u��C�^�                                    Bxs�"  
�          @��?(����\)�\����RC���?(���(������C�
=                                    Bxs��  �          @�=q?+���
=�]p��G�C��?+��'���{C�5�                                    Bxs��n  �          @���&ff����.�R�̣�C�� �&ff�XQ������c�C���                                    Bxs��  �          @�Q�u����#�
�ȣ�C�C׿u��G��c�
���C�                                    Bxs��  �          @��#�
���
?�@���C��\�#�
��  �/\)�ә�C��=                                    Bxs��`  
Z          @�(��#�
�\�#�
��(�C�B��#�
���\�U���HC�q                                    Bxs�	  �          @�������\>�Q�@UC��)��������9�����C�k�                                    Bxs��  �          @�ff>������\�A�C���>����
��ff�/ffC��
                                    Bxs�&R  
�          @ʏ\?�=q��Q�?J=q@��HC�� ?�=q���������C�f                                    Bxs�4�  �          @˅@l(���(�@\)A��
C�AH@l(����
�
=q��(�C��H                                    Bxs�C�  �          @Ϯ@�(��5�@g�BG�C���@�(����?˅Ag�
C���                                    Bxs�RD  �          @�  @dz����@W�A�=qC�|)@dz�����?333@ǮC��                                    Bxs�`�  "          @���@�  ��\)@��A�C��\@�  ����L�Ϳ�C���                                    Bxs�o�  T          @��
@n{����?��A'�C��R@n{��Q��\)�pQ�C���                                    Bxs�~6  �          @ʏ\@G����H?�@���C�� @G���Q��&ff�Ù�C��                                    Bxs��  "          @Ǯ@33��33>���@l(�C��@33���0  ��Q�C���                                    Bxs웂  �          @Ǯ@G����>��?��C���@G���Q��<(�����C�5�                                    Bxs�(  �          @�{?����=�\)?(�C��H?������Dz���(�C��R                                    Bxs��  
�          @�\)?����{�\)��ffC��{?������g
=�{C��                                    Bxs��t  T          @�  ?����=q�����RC�� ?���X������Q�C���                                    Bxs��  
�          @���=�\)����=�\)?0��C�XR=�\)�����HQ���{C�ff                                    Bxs���  
Z          @�{�u�����\)�'
=C��{�u���\�`  �
=qC���                                    Bxs��f  
�          @�G�>���{�L����33C��{>����\�|���=qC���                                    Bxs�  
�          @�zὸQ����H���c33C����Q��\)����8�RC�aH                                    Bxs��  
�          @�녿Q������p����\C����Q��aG�����T33C�q�                                    Bxs�X  
�          @��þ�=q��  ����(�C��q��=q�XQ���
=�^�
C��                                     Bxs�-�  �          @�Q�@������_\)���C�aH@녿������H�}
=C�,�                                    Bxs�<�  
(          @��@��
�:=q�E���33C�g�@��
�Tz������2�C�J=                                    Bxs�KJ  
�          @�{@�녿�p��1�����C��\@�녾W
=�Z=q��C�Ǯ                                    Bxs�Y�  T          @��@���   �8Q���  C�q�@���
=�qG����C�=q                                    Bxs�h�  �          @��
@"�\���R�z����C��H@"�\�a������;�C��f                                    Bxs�w<  
�          @��H@'
=��=q���
�p��C�� @'
=�^�R�\)�*  C�n                                    Bxs��  
�          @�Q�?������׿c�
�33C�Q�?�����(��|���=qC�!H                                    Bxs픈  "          @ƸR?�  ���u�
=C�
?�  �����~{�=qC�.                                    Bxs��.  
�          @�?�G���z����33C�XR?�G���Q����;��C�޸                                    Bxs���  �          @�(�?�G���(���
=���RC���?�G��w�����:��C���                                    Bxs��z  
�          @�z�?��������mp��
=C�  ?��ÿ�{��G���C�@                                     Bxs��   T          @�z�?�33����U��Q�C��?�33�ff�����|��C�e                                    Bxs���  T          @�
=?�����ff�%��  C�  ?����O\)���H�]\)C���                                    Bxs��l  
Z          @��?&ff���H��33�{33C�,�?&ff��=q��  �>=qC��=                                    Bxs��  �          @�\)?�����H�xQ����C��R?����z�����$
=C�g�                                    Bxs�	�  "          @�ff?��R��(��{��=qC�8R?��R�e���
�PG�C���                                    Bxs�^  �          @�\)?������H�
=q��\)C���?����e������K(�C��                                    Bxs�'  %          @�\)?�{��ff��\���C��3?�{�o\)��Q��HffC���                                    Bxs�5�  �          @���@33�c�
�q��&�C�p�@33��  ���H��C�=q                                    Bxs�DP  �          @ʏ\@Q��G������r\)C���@Q�?��\��(�33A�z�                                    Bxs�R�  �          @�
=@G��&ff���R�C��
@G�@\)���d  B>�R                                    Bxs�a�  
�          @�33@z�\���H#�C�h�@z�@2�\��z��[{BI
=                                    Bxs�pB  T          @�G�@@��    ��{�t=qC��R@@��@8����33�?=qB/��                                    Bxs�~�  "          @ʏ\@B�\?��\��\)�gQ�A���@B�\@w
=�p  ��\BO(�                                    Bxs  "          @��?�
=?:�H���L�A�\)?�
=@e�����:�HBv�                                    Bxs�4  �          @�
=?����R��=q8RC�ٚ?�@(�������BIQ�                                    Bxs��  �          @�Q�@{��  ��(�B�C��)@{@�\��Q��j=qB+Q�                                    Bxs  �          @�33@\)��=q����#�C�3@\)@�\��p��r(�B6�H                                    Bxs��&  �          @�{@녿����=q�C��{@�?����(��fB�H                                    Bxs���  T          @˅>�(��[���\)�Wz�C��
>�(��\���\¨ǮC�y�                                    Bxs��r  �          @��H�J=q��=q�����]G�CR\�J=q?��\���g�
C&f                                    Bxs��  "          @�ff�%��  �Dz�����Cr���%�%��Q��\��Ca                                    Bxs��  "          @�
=�(Q������$z�����Cs���(Q��G
=��  �K=qCe�{                                    Bxs�d  �          @�{�-p���
=��p����
Cs���-p��dz���33�6��ChǮ                                    Bxs� 
  T          @����Mp���
=���
����Cnff�Mp��^{��G��)  Cc.                                    Bxs�.�  
�          @ʏ\�z�H��G���  �6�\Cf���z�H�XQ��g
=���C\�
                                    Bxs�=V  
]          @�=q��Q���=q��{�l��Cd� ��Q��@���s33���CX�H                                    Bxs�K�  
�          @θR�dz���\)����#
=Ck���dz��tz��p���Cb��                                    Bxs�Z�  T          @���XQ����R���\�333Cn=q�XQ��|(��\)�33Ce\)                                    Bxs�iH  
]          @�{��z��У�>��@�=qC��쿔z���Q��E���G�C�L�                                    Bxs�w�  
�          @�{�����z�5���
C�׿��������Q��C{�R                                    Bxs  
Z          @��H������Q��1���(�C�ٚ�����J�H��G��effCyB�                                    Bxs�:  �          @�{��ff��ff�z�H�C�AH��ff�����=q=qC�@                                     Bxs��  "          @ə�>\)��G����\�H(�C�H>\)�W
=�Ǯ¤u�C��=                                    Bxsﲆ  �          @ə�?Y���@  �����p�C���?Y��>�33��p�¢��A�33                                    Bxs��,  T          @�
=?�녿�����
=
=C���?��@�����B�u�                                    Bxs���  T          @�(�?�p�>W
=���
z�@أ�?�p�@e������T(�B�z�                                    Bxs��x  
�          @�p�@z�>B�\��\)\@�  @z�@_\)��p��KBa��                                    Bxs��  �          @�ff@
=>�(���  �A#�@
=@l�������D{Be��                                    Bxs���  �          @�G�@�?�z������A�Q�@�@��\��ff�!�RBu�                                    Bxs�
j  �          @�
=@0��@(Q���\)�Wz�B.Q�@0��@����@����{Bt�\                                    Bxs�  �          @ʏ\@ ��@'
=�����g��BQ�
@ ��@����C33�뙚B�ff                                    Bxs�'�  �          @�z�>�
=��������«ǮC�e>�
=@HQ���z��i�B��                                    Bxs�6\  �          @�z�\�L������£33CtW
�\@"�\����{B�=q                                    Bxs�E  �          @�
=��
=���
���W
C\�R��
=@�\��G�#�BꙚ                                    Bxs�S�  "          @�Q��׿!G��\¥B�Ci�����@-p���\)�}Q�Bǀ                                     Bxs�bN  
�          @�
=>��?c�
���
¢ǮB��R>��@�����ff�Ez�B���                                    Bxs�p�  
�          @��?�ff@�\���#�Bz�?�ff@���w
=�\)B��                                    Bxs��  
�          @�\)?�Q�@�������*�B��H?�Q�@�(���ff�?�B��                                    Bxs��@  
�          @���?�\)@��H�#33����B�W
?�\)@\?Tz�@�p�B�#�                                    Bxs��  T          @���?�
=@����o\)��B�� ?�
=@Ǯ������B��=                                    Bxs�  
�          @�(�?�{@����w��(�B��
?�{@Å�8Q��У�B��                                    Bxs�2  "          @�(�@3�
@�ff�dz���\Bf��@3�
@�p��#�
����B~p�                                    Bxs���  �          @��H@l��@S33�y����RB'  @l��@�(����
�`��BS33                                    Bxs��~  �          @�Q�@:=q@=p�����A(�B5�H@:=q@���
=���Bn��                                    Bxs��$            @�(�?O\)@Fff����h�B���?O\)@��
�-p���B��
                                    Bxs���  �          @�33>��@L(�����h��B�Ǯ>��@�ff�*=q��B���                                    Bxs�p  �          @���?:�H@'
=���H�|ffB��H?:�H@���G���G�B���                                    Bxs�  �          @\?��\@c�
��G��QG�B��)?��\@�=q�	�����\B�#�                                    Bxs� �  �          @��
?G�@w
=��z��F�B�=q?G�@�  ��{��ffB���                                    Bxs�/b  
�          @�33?�R@g
=�����Q��B��=?�R@���
=���B�aH                                    Bxs�>  �          @�{@��@N�R���R�C�B_  @��@���33���B�u�                                    Bxs�L�  �          @�?��H@���l(���HB��f?��H@�Q�8Q��׮B��                                     Bxs�[T  �          @ȣ�?���@�(��^�R��
B�k�?���@����{�FffB���                                    Bxs�i�  �          @�z�@'
=@��\�e��Be�@'
=@���Y���G�B�
=                                    Bxs�x�  �          @�=q@8Q�@q��n�R��\BR��@8Q�@�{����-��Bs�H                                    Bxs�F  �          @��@!G�@���;���BmG�@!G�@��;W
=���B�
                                    Bxs��  �          @���@>{@\)�QG��\)BUz�@>{@��
�#�
��z�Bo��                                    Bxs�  �          @�
=@5@H�������/{B?��@5@�=q��  ��33Bn{                                    Bxs�8  �          @��R@{@=q����Y\)B1z�@{@���1���  BwQ�                                    Bxs���  �          @�G�@P  @0  ��\)�3\)B!{@P  @��H�
=���RBZ                                    Bxs�Є  �          @�\)@h��@,���x���"�B=q@h��@��������Q�BH�                                    Bxs��*  �          @��@\(�?�Q���=q�>Q�A�33@\(�@|���'�����BC��                                    Bxs���  �          @��
@_\)?��H���
�A=qA���@_\)@q��1����B=                                      Bxs��v  �          @�\)@w
=?&ff�����:�A�@w
=@1G��J=q���B�\                                    Bxs�  �          @���@L(��8Q������Up�C��q@L(�?�p�����B{A�(�                                    Bxs��  �          @�\)@*=q�.�R��33�<��C��@*=q�����v��C��3                                    Bxs�(h  �          @��@  ������  �xC�+�@  ?��R��
=�v
=B��                                    Bxs�7  �          @�p�?�����
��u�C��?��@ff���H�^��BR�                                    Bxs�E�  �          @���?��R?�
=��=q�\B/Q�?��R@q��Tz���
B��                                    Bxs�TZ  �          @��?�G�?Ǯ��p��}�B&�\?�G�@s�
�G���B��                                     Bxs�c   �          @��H?���@'
=��ff�i  Bi�
?���@�33�333���
B�k�                                    Bxs�q�  �          @���@
=@:�H��  �?�HBXz�@
=@�33��=q����B��                                    Bxs�L  �          @�{@@H���a��,�\Ba{@@�G������l��B�p�                                    Bxs��  
�          @��
@�\@<�������PffBP@�\@���\)��\)B�=q                                    Bxs�  �          @�Q�@$z�@(�����I33B7�@$z�@��H�ff��ffBs33                                    Bxs�>  �          @�p�@�?����\�f33BQ�@�@~{�:�H�  Bp=q                                    Bxs��  �          @�\)?�=q@�
��\)�o(�BA�?�=q@�Q��:=q� ffB��=                                    Bxs�Ɋ  �          @��R?�
=@(���33�e
=B]��?�
=@�Q��&ff���HB�=q                                    Bxs��0  �          @���?���@U��ff�HB���?���@�ff���R���\B��q                                    Bxs���  �          @�G�@(�@�33�l(���
Bl�
@(�@�{�p���  B��{                                    Bxs��|  �          @���@#�
@���O\)���Bn33@#�
@�p���(���=qB�\)                                    Bxs�"  �          @��H@�@`  �xQ��)=qB\��@�@��׿��e�B�L�                                    Bxs��  �          @��@!�@@  �hQ��+{BG33@!�@�
=��p����RBq�
                                    Bxs�!n  Q          @�\)@#33@l(��I���p�B]=q@#33@����5��ffBw��                                    Bxs�0  T          @�@J�H@g
=�A���BB�H@J�H@�p��&ff���HB_\)                                    Bxs�>�  
�          @��@*�H@g��W
=�33BV33@*�H@�33�n{�(�Bt��                                    Bxs�M`  "          @��\@p  @y������z�B8p�@p  @��>��R@AG�BFp�                                    Bxs�\  �          @�=q@���@_\)�z���=qB#@���@�{��G�����B8��                                    Bxs�j�  "          @�(�@g�@q���H��(�B9{@g�@���#�
�ǮBLz�                                    Bxs�yR  �          @�@�G�@~{������\)B+Q�@�G�@���>�@���B7
=                                    Bxs��  T          @���@���@�����H�~ffB9��@���@��?\(�@��B@
=                                    Bxs�  �          @ƸR@��@�  ��=q��G�B7��@��@�=q?333@�G�B@                                      Bxs�D  �          @�ff@��@N{�"�\��(�B
=@��@�=q��G���G�B-��                                    Bxs��  T          @��
@�p�?�z��Fff��\)A��@�p�@P�׿�\)�t��B(�                                    Bxs�  �          @���@�=q@H���E���33B��@�=q@�G��p���G�B:�                                    Bxs��6  �          @\@��R@c�
�!G��ď\B!  @��R@����  ��B7��                                    Bxs���  �          @Å@���@:�H�*=q����B �@���@x�ÿ5��B�R                                    Bxs��  T          @���@��H@P���1G����
B��@��H@�\)�!G���
=B*�
                                    Bxs��(  
�          @ȣ�@�Q�@Y���,���̏\B{@�Q�@�=q���H��Q�B.��                                    Bxs��  �          @��H@�p�@]p��ff��  B�@�p�@��.{����B-�\                                    Bxs�t  �          @�33@|(�@l(������G�B,��@|(�@���=�Q�?\(�B=�                                    Bxs�)  
�          @��@x��@b�\�����
B)=q@x��@��R��\)�(��B=(�                                    