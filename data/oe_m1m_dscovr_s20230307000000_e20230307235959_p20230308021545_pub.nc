CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230307000000_e20230307235959_p20230308021545_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-08T02:15:45.670Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-07T00:00:00.000Z   time_coverage_end         2023-03-07T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxk��@  T          @��@�  ����?���ALz�C���@�  ���H>B�\@  C��
                                    Bxk���  T          @�Q�@~{����?���A���C��{@~{�!G�?(�@�(�C���                                    Bxk���  
�          @�  @�
=���?��
A���C���@�
=��>��H@��
C��=                                    Bxk�2  T          @�z�@q녿�=q?���A��C�{@q�� ��?z�@��C�,�                                    Bxk��  �          @���@_\)����@%B�C��{@_\)�.�R?�{A�Q�C��                                    Bxk�+~  "          @���@U��G�@{A��HC��=@U�\)?�(�A���C��                                    Bxk�:$  
�          @]p�@8Q��
=?�RA((�C��f@8Q��  ��Q����HC�\)                                    Bxk�H�  �          @�p�@y����G�?
=q@�z�C�޸@y����\�����G�C�                                    Bxk�Wp  "          @���>��H@C�
@,��B$(�B���>��H?�
=@z=qB��
B�p�                                    Bxk�f  	i          @�{�xQ�@5�@*�HB'{B��f�xQ�?}p�@p��B�p�CT{                                    Bxk�t�  
�          @��\>8Q�@G
=@<(�B-z�B�G�>8Q�?��@�z�B��B�                                    Bxk��b  T          @���>�=q@C�
@<��B/\)B�Ǯ>�=q?�ff@��
B���B���                                    Bxk��  �          @���>�(�@1G�@Mp�BCz�B�Q�>�(�?(��@�{B���Bc�
                                    Bxk���  T          @�\)>��@G
=@3�
B(�B�aH>��?�z�@���B�{B��{                                    Bxk��T  "          @�
=>\@'
=@P��BK�B�k�>\?   @�z�B���BS��                                    Bxk���  T          @�G�?:�H@5@G�B;(�B�.?:�H?E�@�z�B�G�B:\)                                    Bxk�̠  T          @�33?�z�@A�@8Q�B&�B��?�z�?��@���B���B)��                                    Bxk��F  T          @�z�?   @5�@R�\BC=qB��?   ?.{@�G�B��RBV\)                                    Bxk���  "          @�G�?(��@E�@6ffB(�B��?(��?�\)@���B�L�Bm�                                    Bxk���  �          @�G�?�@�
@b�\B`z�B�{?�=���@�\)B�L�A\)                                    Bxk�8  �          @��?��@�@]p�B^��B33?�����
@��B�.C���                                    Bxk��  �          @�
=?�33?�=q@R�\BM33B033?�33�.{@qG�B|�RC�s3                                    Bxk�$�  
�          @�33?�  @
=q@H��BNz�Bo�?�  >8Q�@s�
B�ffA{                                    Bxk�3*  
Z          @|��?��?�33@Z�HBr�B33?���Q�@`��B~�\C��                                    Bxk�A�  �          @{�?�p�?!G�@Z�HBv��A�=q?�p���G�@O\)Bb=qC�f                                    Bxk�Pv  �          @��\@�\?��
@Mp�BO�RA\@�\�L��@QG�BU�
C�Y�                                    Bxk�_  �          @��?�(�?У�@C33BG�Bff?�(��W
=@\��BpQ�C��                                    Bxk�m�  K          @��\@G�@#�
@�B�BA��@G�?��@L��BP  A�ff                                    Bxk�|h  
�          @��H@Q�@�@�RB{B*Q�@Q�?\)@P��BTz�ATz�                                    Bxk��  �          @�=q@\)?�@7
=B3Q�B  @\)=#�
@X��Bb33?�\)                                    Bxk���  �          @�G�@/\)?��@(�B��B
33@/\)>�@6ffB7=qA�
                                    Bxk��Z  �          @��@>{?�{?��RA�\B ff@>{?��@*�HB%��A&�R                                    Bxk��   �          @���@��?�@E�BM��AC�
@�ÿ�@8��B=�\C��q                                    Bxk�Ŧ  T          @���@
�H@
=@	��B�RB=��@
�H?aG�@Dz�BR�RA�G�                                    Bxk��L  �          @���@ff@3�
?�33A�BT��@ff?�33@EBJ��B�                                    Bxk���  "          @��@�\@%@��B�RBA�H@�\?��@N{BO��A�Q�                                    Bxk��  
�          @��
@{@
�H@�RB�B%z�@{?\)@P  BPffAK�                                    Bxk� >  +          @x��@�?�\@��B{B{@�>��@=p�BI�@�=q                                    Bxk��            @E@�?��
?�
=A��B�@�?0��?�B�
A�ff                                    Bxk��  �          @a�@<��?#�
?�\A�ABff@<�;\?���A�Q�C�L�                                    Bxk�,0  
�          @}p�@$z῞�R@(Q�B*��C��@$z���?���A�Q�C�K�                                    Bxk�:�  
�          @xQ�?��H��G�@1G�B9��C�H?��H�<��?�Q�A��C��                                    Bxk�I|  "          @tz�?��Ϳ�(�@%�B>�RC�}q?����3�
?��A�  C���                                    Bxk�X"  �          @g
=@8Q�?���?\)A�B\)@8Q�?�G�?�
=AÙ�A��                                    Bxk�f�  �          @Z�H@B�\?��\?p��A}�A�\)@B�\?��?��HA��A1                                    Bxk�un  �          @k�@Vff?n{?z�HA{�Axz�@Vff>�z�?���A���@��
                                    Bxk��  �          @l��@]p�����?�  A�{C�G�@]p��n{?c�
A`z�C�xR                                    Bxk���  T          @�33@p�׽u?���A�z�C���@p�׿aG�?��A��RC�c�                                    Bxk��`  T          @��@j=q��G�?�p�A�z�C�.@j=q�}p�?�
=A���C�o\                                    Bxk��  �          @�Q�@qG�����?��\A�{C��)@qG��h��?n{AW
=C�1�                                    Bxk���  �          @�33@l(���p�?�Q�A�{C�*=@l(���Q�?�G�A�p�C�\                                    Bxk��R  �          @�  @aG���Q�@
=BC�
@aG����
?�=qA�  C�AH                                    Bxk���  
          @��@aG�>��@��Bz�@�(�@aG��c�
@��A���C�޸                                    Bxk��  T          @�z�@^�R?W
=@ ��BQ�AX��@^�R�z�@%�B�HC�:�                                    Bxk��D  T          @��@Y��?�@#33B{A
ff@Y���aG�@��Bz�C��                                    Bxk��  �          @��
@s33��z�@�\A�C���@s33���?���A�p�C�|)                                    Bxk��  "          @�z�@\)��?ٙ�A���C���@\)���\?�(�A��
C�*=                                    Bxk�%6  �          @��
@}p��p��?�=qA�{C�T{@}p���\)?^�RA7�
C��)                                    Bxk�3�  "          @��H@~�R�.{?ǮA�
=C�*=@~�R���?}p�AR�\C�c�                                    Bxk�B�  
�          @�=q@z�H��?�p�A��HC��f@z�H��ff?�p�A���C��3                                    Bxk�Q(  �          @�=q@\)�\)?�G�A�=qC�H@\)��G�?�  AW�C�@                                     Bxk�_�  "          @�\)@~{����?��A��\C�%@~{���
?p��AO33C��R                                    Bxk�nt  �          @�@x�ÿ��?(��A��C�~�@x�ÿ��R�����C���                                    Bxk�}  T          @��
@xQ쿮{>�Q�@�Q�C�S3@xQ쿬�;Ǯ��
=C�b�                                    Bxk���  �          @�33@s33��Q�?�\@�33C�� @s33���R���R���C�C�                                    Bxk��f  "          @�
=@z�H����?&ffA�\C�S3@z�H�\����33C�n                                    Bxk��  
�          @��R@|�Ϳ�(�?O\)A1�C�l�@|�Ϳ��H=�\)?fffC�ٚ                                    Bxk���  �          @�G�@o\)�}p�?�ffAs�
C���@o\)���>���@��C��\                                    Bxk��X  
�          @���@r�\�k�?��As\)C�.@r�\��=q>�G�@�ffC�N                                    Bxk���  "          @�Q�@p�׿aG�?��
Ao�C�e@p�׿��>�(�@�
=C���                                    Bxk��  "          @�=q@s33�xQ�?z�HAa��C��@s33���>�{@�z�C�E                                    Bxk��J  T          @��@i����z�?^�RAB�HC�˅@i����\)��G���  C�w
                                    Bxk� �  �          @|��@j=q���
?n{A[\)C�&f@j=q��{>�=q@|(�C�Ф                                    Bxk��  
Z          @~{@o\)�c�
?s33A]��C�Q�@o\)��  >�Q�@�ffC��                                    Bxk�<  
�          @���@tz�Y��?p��AXz�C���@tzῚ�H>�p�@�=qC�1�                                    Bxk�,�  �          @��\@~�R��=q?G�A0  C��@~�R�!G�?�@�{C�}q                                    Bxk�;�  }          @�(�@��þ8Q�?J=qA2{C��
@��ÿ\)?
=A�
C��                                    Bxk�J.  "          @�p�@��׿n{?z�A ��C�t{@��׿���<��
>��
C�W
                                    Bxk�X�  �          @��@��׿L��?+�A  C�c�@��׿��
>.{@33C��
                                    Bxk�gz  �          @��H@�Q�?8Q�\����A"�H@�Q�?O\)=#�
?
=A733                                    Bxk�v   
�          @��
@���>aG��\)����@Dz�@���>�׾�p�����@�(�                                    Bxk���  �          @�Q�@|��>.{�&ff��@p�@|��>������Q�@�(�                                    Bxk��l  T          @�=q@w�>�ff��=q�w�@�z�@w�?p�׿333�33AY                                    Bxk��  �          @�  @^�R?����������A��@^�R?�z�Ǯ����A�\                                    Bxk���  +          @|(�@W�?\��ff���\A���@W�?��H��33����A�                                    Bxk��^  }          @���@B�\?˅�z�����A��@B�\@�H�xQ��`��B=q                                    Bxk��  
�          @��@[�?���G�����A�@[�@�R����
=B�                                    Bxk�ܪ  }          @���@S�
@�Ϳ�=q�r�\Bz�@S�
@(�>B�\@(��B�R                                    Bxk��P  
�          @��\@X��@
�H�   ��\B�@X��@�?+�AQ�B G�                                    Bxk���  
�          @�Q�@7�@)���Tz��@Q�B+  @7�@-p�?!G�A��B-ff                                    Bxk��  �          @~�R@B�\@�H����BG�@B�\@Q�?=p�A-G�B(�                                    Bxk�B  T          @���@7
=@+��O\)�:ffB,��@7
=@.{?(��A  B.Q�                                    Bxk�%�  "          @��@J�H@#�
������B��@J�H@{?Y��A>{B�                                    Bxk�4�  
�          @�@H��@'��J=q�.ffB�@H��@*=q?&ffA�HB!(�                                    Bxk�C4  "          @�{@R�\@{�333�B@R�\@\)?&ffA�HBQ�                                    Bxk�Q�  
�          @�z�@l��@ff����[\)A�z�@l��@�>#�
@��B
=                                    Bxk�`�  
�          @���@X��>�33�����  @�p�@X��?��׿��R��G�A��
                                    Bxk�o&  T          @�z��{��=q?
=q@�(�Cz�{��{��
=������Cy�                                    Bxk�}�  T          @�ff��33�����#�
��{C�E��33��=q�#�
��p�C~8R                                    Bxk��r  
(          @�G�������z�>�?���C�y�������G��ff��p�C33                                    Bxk��  "          @��
�\���
�����Z=qC|�f�\����-p�� =qCyp�                                    Bxk���  "          @��\��  ����������Cy� ��  �w
=�1��33Cu�f                                    Bxk��d  �          @�����Q�������\���HC}E��Q��x���6ff�
  Cy��                                    Bxk��
  
�          @��׿��H��33�W
=���C����H���H�'
=��{C}��                                    Bxk�հ  K          @�(���33��
=��z��\��C�%��33�|(��'��Q�C}��                                    Bxk��V  
�          @��ÿ�
=��\)�E��z�C�ῗ
=�mp��B�\�{C|Q�                                    Bxk���  T          @�녿��\���׿����P  C������\�XQ��L(��(p�C}.                                    Bxk��  "          @�=q�������>�z�@Z�HCwLͿ����~{���H��33Cu�                                    Bxk�H  T          @��ÿ�{��p��!G���(�Cz��{�aG��0���  Cuk�                                    Bxk��  	�          @�������{=�Q�?�z�C��þ���x���������C���                                    Bxk�-�  �          @�G��\)���R�������C�{�\)�j=q�%��C��H                                    Bxk�<:  T          @��R>����\�����C���>��]p��)����C��                                    Bxk�J�  
�          @��=�\)���׿s33�Up�C�xR=�\)�@���5��,�HC��H                                    Bxk�Y�  �          @�?��H�^�R��\)�zffC��R?��H�p��,���"��C�K�                                    Bxk�h,  
�          @�33@C�
�4z῏\)�p(�C���@C�
������z�C�+�                                    Bxk�v�  
�          @��R@|(����H�����{C�G�@|(���  ��  ���HC�1�                                    Bxk��x  �          @���@�����=q=�?�ffC�ff@������H�����\C��                                    Bxk��  +          @�z�@�  ���>�\)@Y��C�^�@�  �����z��c�
C�c�                                    Bxk���            @�=q@��ýL��?�@�(�C���@��þ���>��@��C��                                    Bxk��j  �          @��@p  ��?xQ�AD��C�Ф@p  �"�\��{���C��                                    Bxk��  ^          @��R@z�H��p�?uAIG�C��@z�H��(�������C��H                                    Bxk�ζ  |          @�
=@u����?�  A\Q�C�XR@u���z�>L��@0  C�J=                                    Bxk��\  �          @�Q�@mp���p�?(��AC��\@mp��33�������C���                                    Bxk��  �          @�@���>\?&ffA
{@�z�@���<#�
?B�\A z�>.{                                    Bxk���  ^          @���@��
?=p�?Q�A'�
A�@��
>u?�=qA]G�@E                                    Bxk�	N            @��\@�����Q�>aG�@5�C�q�@����B�\>��?�z�C���                                    Bxk��  
�          @�(�@�Q�>�33?z�HAC\)@��@�Q�.{?��
AL��C��f                                    Bxk�&�  T          @���@��;B�\?�Q�A�\)C��3@��Ϳ�  ?���A�p�C�>�                                    Bxk�5@  �          @���@�z�8Q�?�(�A��RC���@�z῁G�?�z�A��\C�(�                                    Bxk�C�  T          @�
=@���?��?�A�ffA^�\@���=���?�(�AƏ\?���                                    Bxk�R�  
2          @�@r�\?�p�@Q�A�ffA�{@r�\��Q�@*�HB��C�N                                    Bxk�a2  
          @��@g
=?�\)@p�A�(�A��\@g
==�G�@&ffB�?�ff                                    Bxk�o�  "          @��R@{�?fff?���A�p�AM�@{���@�
A���C�R                                    Bxk�~~  T          @��@hQ�?���@\)A�(�A�(�@hQ�>.{@*=qB��@,(�                                    Bxk��$  
(          @�33@�(�?�\?�Q�A�z�@�\@�(��   ?���A���C��f                                    Bxk���  
�          @�G�@Tz���\?�p�A�  C��
@Tz��5>Ǯ@���C��
                                    Bxk��p  	�          @���@W
=��p�@!G�B�C��=@W
=�<��?��RAw33C�Z�                                    Bxk��  "          @��@^�R���H@"�\B  C��@^�R�.�R?���A���C��                                    Bxk�Ǽ  
�          @�z�@a녾��@:�HB�C���@a녿�ff@�RBC�&f                                    Bxk��b  
�          @�(�@^�R��p�@W
=B/\)C���@^�R���H@0��B
\)C�L�                                    Bxk��  T          @�\)@U���@h��B<p�C�!H@U��G�@9��B  C�޸                                    Bxk��  "          @�z�@l�;��@H��B ��C���@l�Ϳ��@#33A�  C��                                     Bxk�T  
�          @�@�  ���
@5B�C��)@�  ��33@�A�=qC���                                    Bxk��  �          @���@�p���  @	��A�Q�C��@�p����?��
As�C�n                                    Bxk��  �          @�=q@�{����?�A��C��{@�{���?0��A�
C�"�                                    Bxk�.F  �          @�p�@h���#�
?��\A~=qC�s3@h���6ff������C���                                    Bxk�<�  T          @�@aG��0��?�33A���C��R@aG��E��G�����C�]q                                    Bxk�K�  �          @��@l���#�
?
=q@��
C���@l���   �B�\��RC��
                                    Bxk�Z8  "          @�
=@u��      <��
C��\@u���(���=q�a��C�ff                                    Bxk�h�  T          @�=q@H���3�
?��Ab�RC��@H���>�R��
=���\C�AH                                    Bxk�w�  "          @�@p��2�\@*=qBp�C��{@p��n�R?}p�AHz�C��
                                    Bxk��*  �          @�  >aG��&ff@y��B`C�l�>aG�����@
�HA�{C��                                    Bxk���  "          @�z�@ff�qG�@�RA�z�C���@ff��(�>\)?�
=C�˅                                    Bxk��v  "          @�z�@���|(�?�A�p�C��
@�����H��\)�P��C�q�                                    Bxk��  "          @�p�@����
?�A�ffC���@������\���RC��3                                    Bxk���  
�          @��
@��~�R?���A�z�C���@����
���R�j=qC���                                    Bxk��h  T          @��?�ff��\)?(��@��C���?�ff��  �Ǯ����C�t{                                    Bxk��  �          @�p�?��H���\�W
=�(�C�:�?��H�{������C���                                    Bxk��  "          @�33?�{��������g
=C��?�{�s33�=q��  C��H                                    Bxk��Z  "          @�  ?У����=�G�?�{C�"�?У��{��   ���C�AH                                    Bxk�
   	�          @��R?�G���{>��R@g
=C��?�G����׿��H��G�C��
                                    Bxk��  T          @�\)?�Q���G���G���33C��)?�Q���Q��*�H�=qC�J=                                    Bxk�'L  �          @���?��\����333���C��?��\�|(��:�H���C�@                                     Bxk�5�  �          @�(�?u���Ϳ�G��?\)C��\?u�g��Fff�ffC�g�                                    Bxk�D�  
�          @��\@   �\)?5A��C�T{@   �vff���\��(�C���                                    Bxk�S>  
�          @��H@33�Y��?�(�A̸RC��@33�{�=�?�G�C�0�                                    Bxk�a�  �          @��R?��w�?��RA��\C��3?���  �@  �C�Z�                                    Bxk�p�  �          @��\?�=q��ff�k��?\)C�0�?�=q�QG��2�\�ffC�                                      Bxk�0  
�          @�33?����׿��H��  C�  ?��.�R�Z=q�C�\C���                                    Bxk���  
�          @���?s33�s33������C�H?s33��k��^{C�                                    Bxk��|  
�          @��?(���x���33��{C��
?(����w
=�g  C��f                                    Bxk��"  "          @���>.{�c33�=p��Q�C�]q>.{��p������3C�Ǯ                                    Bxk���  
�          @��>u�n{����33C���>u�Q��z=q�u
=C�0�                                    Bxk��n  "          @�{?333��z��G���G�C�˅?333�-p��n�R�TffC�=q                                    Bxk��  
�          @�(�?O\)���Ϳ���YC��3?O\)�dz��L(��$G�C�ff                                    Bxk��  "          @�z�?�����  ��\)�J=qC�U�?�����������C�0�                                    Bxk��`  T          @��?=p���=q���R�s33C���?=p��y���=q��z�C�b�                                    Bxk�  
�          @����=q���\���H��(�C{����=q�,(��h���I�
Cs��                                    Bxk��  
�          @�(���{��
=��\)��\)C5ÿ�{�E�K��0G�Cz.                                    Bxk� R  �          @��@1��AG�?�p�A�\)C�N@1��O\)���
���C�H�                                    Bxk�.�  "          @��@E��0��@G�A��
C�R@E��XQ�?   @�
=C�0�                                    Bxk�=�  T          @�(�@8Q��I��?�G�A�33C�4{@8Q��g
==�Q�?���C�K�                                    Bxk�LD  �          @���@=p��L(�?���Ah(�C�l�@=p��Vff���H��ffC��)                                    Bxk�Z�  T          @��@33�aG�?�z�Av=qC��=@33�j=q�(�� ��C�                                    Bxk�i�  
�          @��
@]p����H@%�BffC�Z�@]p��G�?�(�A���C�aH                                    Bxk�x6  
�          @��H@*=q����=��
?z�HC���@*=q�fff�����=qC�:�                                    Bxk���  
�          @�z�?�Q���{�   ��(�C��=?�Q��mp��   ����C���                                    Bxk���  T          @��R?�Q���
=�:�H�(�C�� ?�Q��hQ��-p���\C�R                                    Bxk��(  T          @�p�@�H��  �����C��)@�H�5�Mp��"�
C�@                                     Bxk���  "          @���@{�qG���\��  C�9�@{�!��S33�2  C���                                    Bxk��t  
�          @�?�ff�dz�� ���Q�C���?�ff�G��xQ��c
=C��q                                    Bxk��  �          @�{?fff�g��'��=qC���?fff� ���\)�t\)C�{                                    Bxk���  �          @�  ?��
�|(���\)��C���?��
�7��A��+��C��                                    Bxk��f  
�          @�\)?���\)��\)���HC�C�?���333�QG��C  C�f                                    Bxk��  �          @��>�33�z=q�B�\�.�RC��R>�33�HQ��p��=qC�=q                                    Bxk�
�  T          @���?
=����W
=�7
=C�?
=�fff���ffC���                                    Bxk�X  T          @��
>Ǯ���׿�=q���\C��q>Ǯ�7
=�O\)�A\)C��R                                    Bxk�'�  T          @����Ǯ�XQ��?\)�$��C��{�Ǯ��33��ff�C�Q�                                    Bxk�6�  T          @��
�W
=�^�R�{�
Q�C�0��W
=��(��r�\�q��Cv�)                                    Bxk�EJ  T          @�=q�����L���\)��Cw{���Ϳ��H�j�H�m
=Cg��                                    Bxk�S�  "          @�Q켣�
�3�
�aG����HC��׼��
�ff��
�1��C��\                                    Bxk�b�  T          @|(�?����,(���  ��G�C��?�����Q����$�\C�S3                                    Bxk�q<  T          @w�?u�&ff�aG�� C���?u?��\�[��B:p�                                    Bxk��  ^          @|��?��R��33�Q��*�RC�%?��R�Ǯ�7��[(�C�j=                                    Bxk���  J          @���@`  � �׾�p���
=C��@`  ��\)���R��=qC���                                    Bxk��.  T          @��\@5�6ff?�AG�C�u�@5�333�G��0z�C���                                    Bxk���  �          @\��@p��z�?�  A�33C��
@p��(Q�=�\)?��RC��                                    Bxk��z  �          @y��@Mp���R?�A��HC���@Mp���?�
=A�G�C�3                                    Bxk��   �          @tz�@G��\)@ ��B=qC��)@G����
?�p�A���C�޸                                    Bxk���  �          @dz�@HQ�>�(�?��A���@��\@HQ�B�\?��A�33C�=q                                    Bxk��l  �          @��
@j�H?�
=?��HA��A��R@j�H?�\?�\)A�Q�@�                                    Bxk��  
Z          @���@   @a�>���@�Br�@   @@��?��A�(�Ba�R                                    Bxk��  ,          @�Q�@0  @#�
?.{A$(�B,{@0  ?��R?�  A�
=B�                                    Bxk�^  
�          @��@+�?��H?�
=A�=qB�@+�?�p�?�
=B�A��                                    Bxk�!  
�          @��R@&ff@&ff@33A�\)B4  @&ff?�z�@?\)B5�\A�                                    Bxk�/�  �          @�p�@1G�@7
=?��
A�{B7��@1G�?��@)��B33Bp�                                    Bxk�>P  
�          @��@E�@8��?�=qAe��B,��@E�@ff@  A�
=B	\)                                    Bxk�L�  
          @�
=@u?�z�?�z�As33A�=q@u?���?��A�  A���                                    Bxk�[�  T          @��\@��?�33?��A���A��H@��?fff?��AîAH��                                    Bxk�jB  �          @��
@���@ff?333A�A�z�@���?���?ǮA��A�z�                                    Bxk�x�  �          @��@u�@   ?��A�G�Aܣ�@u�?�(�@33A�A�p�                                    Bxk���  �          @��R@1�?�G�@EB1=qA��@1논#�
@\(�BL�C�޸                                    Bxk��4  
�          @���?˅@�@aG�BN�
B\��?˅>��H@�p�B�� A�p�                                    Bxk���  �          @�=q@�
@�\@G�B/�B2�H@�
?+�@s�
Bg
=A�{                                    Bxk���  �          @�\)@8Q�?�Q�@1�B�A�33@8Q�>�z�@P  B@�H@�G�                                    Bxk��&  T          @�33@C�
?O\)@S�
B9=qAm�@C�
�:�H@U�B:�\C�G�                                    Bxk���  �          @�
=@HQ�?��@E�B0\)A=q@HQ�\(�@@��B+(�C�L�                                    Bxk��r  T          @��@b�\����@��B
��C�]q@b�\��\)@�
A�Q�C�h�                                    Bxk��  "          @��
@a�?ٙ�?��HA��HAͅ@a�?G�@�B�AFff                                    Bxk���  "          @��@Y��?\(�@{BffAb�H@Y���L��@�Bz�C�Q�                                    Bxk�d  T          @��@p��?�ff?�RA=qA���@p��?fff?���A�Q�AW�                                    Bxk�
  T          @�\)@|(�?�Q�+��  A��\@|(�?�{���
��33A���                                    Bxk�(�  T          @�33@���>�\)>�{@��@s33@���=�G�>�(�@�ff?��H                                    Bxk�7V  
(          @�(�@��\���H�#�
��Q�C���@��\��G��aG��7�C��                                    Bxk�E�  T          @�33@�zῠ  �u�\(�C��
@�zῌ�Ϳ���   C���                                    Bxk�T�  
�          @��\@`���33�0�����C�e@`�׿�\��{����C���                                    Bxk�cH  T          @���@R�\�.{���˅C�8R@R�\������
��{C��{                                    Bxk�q�  
�          @�\)@0  �@��?=p�A%�C�>�@0  �A녿!G��(�C�                                      Bxk���  
Z          @�{@<(���z�@�Bz�C�=q@<(��z�?��A��
C�n                                    Bxk��:  
�          @��H@�?�
=@�BQ�B  @�?(�@&ffBC�\A{�                                    Bxk���  
�          @_\)@�>���?�33B#=qA\)@���?�{B�C�O\                                    Bxk���  T          @   >�33@=q=�@;�B�k�>�33@�?�33A��HB�33                                    Bxk��,  
�          @fff=u@b�\?��Ap�B��=u@<(�@�
B{B��                                    Bxk���  T          @^�R?G�@K�>�
=@陚B�ff?G�@,(�?޸RA��HB���                                    Bxk��x  "          @{�@Q�@2�\?�(�A��BF
=@Q�@   @33B��B                                       Bxk��  
�          @���@\)@G�?k�AO�
BM�@\)@�@
�HA�p�B1�                                    Bxk���  
�          @�Q�@=p�@G�?���A�33B�@=p�?��H@{B�
A�{                                    Bxk�j  �          @���@<��@7
=?�A}B0=q@<��@@G�B ��B                                      Bxk�  
�          @��R@E�@�?�  A�\)Bff@E�?�(�@B	��A�=q                                    Bxk�!�  
�          @�
=@S33@ff��\)���\B�H@S33@
=q?n{AU�B��                                    Bxk�0\  T          @��@L(�@�R>��@��B�\@L(�@z�?�A�ffB{                                    Bxk�?  �          @�  @W�@�\������z�A��@W�@��Ǯ���\B�                                    Bxk�M�  	�          @���@x��?�z�aG��@  A�@x��?�33�L���,��A�(�                                    Bxk�\N  
�          @��@Mp�@(�?��A���B	�\@Mp�?�ff@=qB��A�(�                                    Bxk�j�  -          @��@|(�?��
?�RAffA�{@|(�?���?��HA��
A33                                    Bxk�y�  
�          @��@z�H?˅��(����\A�
=@z�H?У�>�\)@qG�A�ff                                    Bxk��@  �          @���@�
=?(��?W
=A2�\AG�@�
=>u?��A]G�@P��                                    Bxk���  T          @�G�@u�?��þ�\)��Q�A�z�@u�?���>���@�ffA�(�                                    Bxk���  
�          @�
=@q�?�{��������A���@q�?�{�Y���;\)A��H                                    Bxk��2  T          @�@���?L�Ϳ8Q���RA4Q�@���?��
���
��33Aep�                                    Bxk���  �          @�{@xQ�:�H�&ff��C���@xQ�Ǯ�fff�O\)C�q                                    Bxk��~  �          @�ff@Tz��{�˅����C�j=@Tzῂ�\�{�G�C�u�                                    Bxk��$  T          @��\@`  �˅��Q���  C��f@`  �#�
���	�C��3                                    Bxk���  T          @��\@��l��=L��?�RC�AH@��XQ��  ��33C�e                                    Bxk��p  �          @�Q�@.�R�K���G���p�C�O\@.�R�6ff����Q�C��H                                    Bxk�  �          @��@Vff���?+�A�HC��@Vff� �׾����G�C���                                    Bxk��  "          @���@J=q�5>��R@��C�  @J=q�-p��fff�C
=C���                                    Bxk�)b  �          @�\)@G
=�6ff�.{�ffC�˅@G
=�!G���=q��\)C���                                    Bxk�8  
�          @��@QG�� �׼#�
�.{C�B�@QG��녿�ff�k33C���                                    Bxk�F�  
�          @���@e� ��>��R@��\C�g�@e���H�
=q���C���                                    Bxk�UT  
�          @��
@vff���
?�RA
�\C���@vff��=#�
?�C��f                                    Bxk�c�  T          @��@u���
?��Aw�C�h�@u��\?p��AY��C�9�                                    Bxk�r�  
�          @�
=@`  ���>��@��C���@`  ��R�����33C���                                    Bxk��F  �          @�33@.�R�333��  �fffC�q@.�R�p���\)���C�                                      Bxk���  �          @��\�\)�:=q�Dz��8{C��
�\)��33��  ��CxT{                                    Bxk���  �          @�33��G��U��0����C�ͽ�G���z��xQ��~��C�XR                                    Bxk��8  "          @�(�>�Q��i�����z�C��3>�Q����j=q�b��C�W
                                    Bxk���  �          @�33>Ǯ�q녿�p��ۮC���>Ǯ�)���U�M(�C�/\                                    Bxk�ʄ  �          @�Q�B�\�;��<(��0�C���B�\���R�w�k�Cr�q                                    Bxk��*  T          @��þ�{�c33��=qC�L;�{��\�e��dG�C��
                                    Bxk���  
�          @��H�z��j=q�����{C�z�z���R�\(��U�C�c�                                    Bxk��v  
�          @��R�
=�E�0  �$p�C��{�
=��(��qG�u�Cz��                                    Bxk�  
�          @��׿���R�\�)����
C�>���Ϳ�Q��p  �v�C~#�                                    Bxk��  "          @��?���y���c�
�K�C��?���Mp������
C��                                    Bxk�"h  "          @\)?Tz��n{?8Q�A+�C�O\?Tz��l(��W
=�G
=C�Y�                                    Bxk�1  �          @u?Y���^{?��
A��C���?Y���l(��aG��W
=C�}q                                    Bxk�?�  
�          @xQ�?+��j=q?\(�AO�C�33?+��l�Ϳ.{�$z�C�&f                                    Bxk�NZ  �          @u�?s33�^{?��RA�  C��f?s33�j�H��  �tz�C�=q                                    Bxk�]   "          @Vff>���Dz�?z�A*�RC�j=>���C33�333�M��C�l�                                    Bxk�k�  �          @Vff�5�:=q������ffC�R�5��\)�A��C{�                                    Bxk�zL  �          @�Q쿳33�XQ��'��ffCwuÿ�33��
�p  �a��Ck��                                    Bxk���  "          @�  ��
=���J�H�8\)Cf���
=�h���vff�t{CMG�                                    Bxk���  �          @�{�\)��(��[��L=qCN0��\)>u�h���]��C.�                                    Bxk��>  �          @�ff��
�Y���l(��a=qCHG���
?.{�n{�d��C#��                                    Bxk���  
�          @����33����l���]��CMh��33>��u��j
=C(��                                    Bxk�Ê  
�          @�(����{�|���lp�C<:��?�  �qG��[Q�C��                                    Bxk��0  "          @���z�>B�\��  �o�C/\)�z�?�  �g
=�LQ�C�q                                    Bxk���  _          @���\)��\)�xQ��d��CN� �\)?   ��Q��p�C'n                                    Bxk��|  I          @��{�ٙ��dz��H(�CV�)�{�#�
�}p��h
=C7��                                    Bxk��"  T          @�p��p��333�0���ffCd��p����R�h���N�
CS.                                    Bxk��  �          @��R�7
=��
�C33�#��CW�7
=�333�g
=�K\)CA�                                     Bxk�n  �          @�
=�%�(���4z��z�Ca���%�����hQ��M33CO
                                    Bxk�*  �          @�{���<(��J�H�,��CpaH����(�����u33C]Q�                                    Bxk�8�  "          @�ff��  �AG��O\)�/33Cs����  �\����{{CaL�                                    Bxk�G`  T          @����\)���qG��]�Cgk���\)�������33CB
=                                    Bxk�V  �          @��׿�p��S�
�A��G�Cu녿�p������=q�n\)Cg�                                     Bxk�d�  �          @�=q��Q��mp��2�\�ffC|:ῘQ����Q��c33Cs\                                    Bxk�sR  {          @�=q�˅�Q��HQ��"�HCt0��˅������oCd�=                                    Bxk���  �          @�{�ff���Q��:\)CcW
�ff�W
=�z=q�o�CI��                                    Bxk���  
�          @��?�G���(��\(��+\)C�?�G��mp���R�z�C�j=                                    Bxk��D  "          @���?��\��\)�\)��  C�p�?��\�z�H�  ��{C�S3                                    Bxk���  �          @�Q�?Ǯ��{��(���Q�C���?Ǯ�{��
=�ӮC��{                                    Bxk���  �          @�ff?�=q�}p�?���A�  C��)?�=q���
>B�\@Q�C��                                    Bxk��6  �          @��?�(��Z�H@%�B��C��)?�(���z�?�=qAS
=C��
                                    Bxk���  
�          @�  @J=q�H��?�(�At��C��{@J=q�W����
�n{C���                                    Bxk��  
�          @�\)@7
=�a�?���Al��C��=@7
=�n{�u�8��C��                                    Bxk��(  
�          @��R@8���aG�?��AW�C���@8���j�H�������C�)                                    Bxk��  T          @�ff@&ff�i��?��@�z�C��R@&ff�fff�Tz��)��C���                                    Bxk�t  
�          @��@1G��l(������i�C�q�@1G��<��� �����\C���                                    Bxk�#  �          @�
=@%��]p�?޸RA�33C�P�@%��w
=>�z�@g�C��R                                    Bxk�1�  "          @��@<���aG�?��A_�C�H@<���l(���=q�S�
C�U�                                    Bxk�@f  
�          @��@!��a�@�Aȏ\C�˅@!�����?��@��C�f                                    Bxk�O  T          @���@\)�3�
@.{B�C�Ǯ@\)�hQ�?��HA�\)C�:�                                    Bxk�]�  �          @��@�
��@L��B2p�C��
@�
�S�
@
=A܏\C�q�                                    Bxk�lX  
�          @���?����8Q�@<(�B"�RC�O\?����q�?�33A��HC�                                    Bxk�z�  
�          @�33@&ff�E�@
�HA���C�
@&ff�j�H?\(�A-��C���                                    Bxk���  �          @�
=@9���E�@G�A�  C��H@9���g�?8Q�A�C�aH                                    Bxk��J  �          @��@(Q��Fff@�A�C�.@(Q��p��?��AK�
C���                                    Bxk���  
�          @�{@���'�@J=qB)��C�e@���g
=?��HAŮC�f                                    Bxk���  �          @�  @
=q�p�@W
=B7  C���@
=q�a�@��A�  C���                                    Bxk��<  
�          @�\)?�(��`  @2�\Bz�C�ff?�(�����?��A\)C�xR                                    Bxk���  "          @��H?�(����\?�G�A��C��?�(���33��G���
=C�W
                                    Bxk��  �          @��?����\)?��A��\C��H?�������
����C�'�                                    Bxk��.  �          @�
=?����p�?��AW�
C��?����Q��R���HC��                                    Bxk���  �          @��?n{��  ?G�A{C�޸?n{��
=�n{�8��C���                                    Bxk�z  �          @���?(������?��@�C�'�?(����ff��{�^{C�>�                                    Bxk�   
�          @��
?����\    <��
C�H�?���Q��
=���C���                                    Bxk�*�  
�          @���?
=��\)�����u�C��H?
=��녿�
=��33C�#�                                    Bxk�9l  �          @���?!G�����z���G�C�W
?!G��j=q��
��  C��                                    Bxk�H  T          @��R?�����?��A���C�XR?���%�>Ǯ@��C��                                     Bxk�V�  �          @���@ff>�{@eBmQ�A��@ff�}p�@]p�B`C�e                                    Bxk�e^  �          @���@33?���@^�RB^A�p�@33���@k�Bs\)C���                                    Bxk�t  �          @�{?�33=���@u�B�ff@Tz�?�33��ff@fffBo
=C���                                    Bxk���  �          @�z�?��?@  @hQ�Bw\)A��?���(�@i��Bz�\C���                                    Bxk��P  �          @��
@�R?��@Dz�BBp�A�{@�R����@P��BR�C��                                    Bxk���  T          @��\@   ?��
@C�
BB(�A�z�@   ���@N{BP��C�K�                                    Bxk���  �          @�Q�@�R?��@C33BF��A��@�R>�@Tz�B`(�@Mp�                                    Bxk��B  �          @�Q�?���?���@;�B=��B9�\?���?B�\@[�BnA���                                    Bxk���  
�          @��\?���?�\)@EBFp�B4�
?���?!G�@c33Bt�A�Q�                                    Bxk�ڎ  �          @��
@��?���@>{B:(�B!@��?#�
@Z�HBc�\A��R                                    Bxk��4  "          @���?�ff@�\@;�B4�\BO=q?�ff?��@c�
BmffA�G�                                    Bxk���  T          @�p�?���@333@*�HB!{Bz�?���?�z�@_\)Bg  BB��                                    Bxk��  T          @��?���@ff@333B7�BpG�?���?���@]p�By��B%�H                                    Bxk�&  
�          @o\)?�33?�=q@AG�BY��Bf�H?�33?!G�@^{B�z�A�                                    Bxk�#�  �          @q�?�Q�@�\@B�BV\)?�Q�?��@@��BZffB��                                    Bxk�2r  T          @xQ�@\)?s33@0��B8G�A��@\)��Q�@:�HBF=qC���                                    Bxk�A  
�          @�=q@E��u@%B {C���@E����@B��C��
                                    Bxk�O�  �          @�Q�@
=��Q�@�B	�C�Ff@
=�#33?�G�A���C�`                                     Bxk�^d  T          @�G�?����Vff?���Ayp�C�t{?����`�׾8Q��(��C���                                    Bxk�m
  T          @�ff@(���Dz�?���Alz�C�U�@(���P  ���
��\)C���                                    Bxk�{�  T          @y��@'�?&ff@$z�B.z�A^=q@'�����@(Q�B3�C�                                    Bxk��V  
�          @l(�@���=q?��A���C�˅@���  ?n{A�(�C��                                     Bxk���  T          @:=q?����  ?L��A���C�k�?�����ý#�
�W
=C���                                    Bxk���  T          @G�?=p���\)>#�
@�
=C�޸?=p���ff��\�hQ�C�5�                                    Bxk��H  �          @p�?p���(�=�Q�@z�C���?p����.{��33C�"�                                    Bxk���  
�          @-p�?�\)?���?E�A��BI�?�\)?Ǯ?�33B   B033                                    Bxk�Ӕ  
�          @�33?�
=@XQ�?��RA��
Bp�?�
=@.�R@B��B[
=                                    Bxk��:  T          @��
?�33@dz�?E�A-��Bw�
?�33@E�?��HA�{Bi=q                                    Bxk���  �          @��?�\@Tz�?���A�Q�Bw?�\@'
=@ ��B�HB_��                                    Bxk���  
�          @��\@��@>�R@�\A��BYG�@��@   @K�B=Q�B,�                                    Bxk�,  
�          @�z�@,��?�(�@#33B\)B��@,��?:�H@?\)B;�Ar�H                                    Bxk��  
�          @��@2�\@@B�B (�@2�\?�33@@  B/�\AԸR                                    Bxk�+x  
�          @��R@:�H?�=q@!�B��A�p�@:�H?�R@:�HB1=qA?
=                                    Bxk�:  �          @�(�@N{?�G�@Q�A��RA��@N{>�@�B(�A�                                    Bxk�H�  �          @���@:=q@1녿\)��
=B.�R@:=q@3�
>�
=@�
=B/                                    Bxk�Wj  �          @���@\)?�(��&ff�'�RA�  @\)@G���Q���B)33                                    Bxk�f  �          @i��?�G�?�
=�5��T��B?�G�@z�������BFG�                                    Bxk�t�  �          @mp�@
=@33����z�B$33@
=@%��{��ffB>z�                                    Bxk��\  �          @^{@5�?�녿����{A�G�@5�?�
=���	G�B	p�                                    Bxk��  �          @\��@!�?�=q��\)��{B p�@!�@�\�}p���33B�R                                    Bxk���  �          @7
=@Q�?Y�����\��Q�A�@Q�?�p��h�����\A��                                    Bxk��N  �          @c�
@��?���"�\�:p�A�\)@��?���G���B%=q                                    Bxk���  �          @i��@-p�?^�R����A���@-p�?�
=��z���p�Aޏ\                                    Bxk�̚  �          @|��@:=q?���  ��(�A��@:=q@33����yB(�                                    Bxk��@  .          @r�\@(�@\)���\��Q�B5��@(�@1G���Q���33BB=q                                    Bxk���            @�Q�@�@p����㙚B:G�@�@<(��h���Y�BN33                                    Bxk���  �          @�  @(Q�@  ����㙚B"33@(Q�@0  ���\�n�HB9                                      Bxk�2  �          @y��@!G�@���\���B�@!G�@*�H���H��{B:�                                    Bxk��  �          @vff@L(�>W
=���p�@q�@L(�?^�R�����Au�                                    Bxk�$~  "          @tz�@HQ�\(���p���p�C�N@HQ����
=q�
33C���                                    Bxk�3$  T          @s�
@Z=q�����
��
=C��{@Z=q<#�
��\)����=�                                    Bxk�A�  �          @w�@c�
����
=��p�C���@c�
>�ff��{��ff@�(�                                    Bxk�Pp  �          @s33@S33?!G���  ��Q�A,��@S33?���Q���G�A��\                                    Bxk�_  T          @q�@1G�?��R�p��G�A���@1G�?�33��
=��=qB	�
                                    Bxk�m�  T          @h��?����?\)A2�\C���?����þu��{C�<)                                    Bxk�|b  
Z          @_\)?�  �C�
?�z�A��
C�
=?�  �QG�=��
?�33C�~�                                    Bxk��  
Z          @W�?�(��{?�33A�z�C�k�?�(��-p�>�=q@���C�33                                    Bxk���  "          @`  @C33�˅=���?�\)C�9�@C33�����(���p�C���                                    Bxk��T  
�          @n{@HQ���
?(�A��C�'�@HQ��׼#�
�#�
C�xR                                    Bxk���  "          @s33@E���
?B�\A:ffC�#�@E����=L��?B�\C�G�                                    Bxk�Š  T          @qG�@,����?aG�AY��C��@,���%�=#�
?��C�"�                                    Bxk��F  
�          @qG�@%� ��?fffA`z�C���@%�*�H=#�
?\)C�)                                    Bxk���  
�          @s�
@(��=p�?}p�Ap(�C�<)@(��G������C��                                    Bxk��  T          @tz�@)����?��A�C�` @)���(��?�\@�Q�C���                                    Bxk� 8  �          @r�\@2�\�  ?�z�A��C��3@2�\�   >�p�@���C�\                                    Bxk��  "          @p  @`  �녿!G��"{C�` @`  ��{�G��H��C�C�                                    Bxk��  T          @p��@k���\��=q���HC��@k����;�
=��p�C���                                    Bxk�,*  �          @n{@g
=�Q녾B�\�7�C���@g
=�8Q��G��ٙ�C�ff                                    Bxk�:�  T          @o\)@Vff��녽��Ϳ�=qC��@Vff�\��R�=qC��H                                    Bxk�Iv  �          @mp�@S33��33��\)��C���@S33���R�5�6�\C���                                    Bxk�X  T          @p  @QG���>��@ffC��=@QG���G���(���z�C��R                                    Bxk�f�  �          @p  @Tz��p�>�@   C�:�@Tz��
=��(���=qC��\                                    Bxk�uh  �          @qG�@8���ff>�G�@��C�xR@8������{��(�C�`                                     Bxk��  
Z          @q�@=p��(����H���C��
@=p�����z����C�xR                                    Bxk���  �          @vff@(���/\)���ڏ\C��
@(����Ϳ�ff����C���                                    Bxk��Z  
�          @z=q@=p���ÿW
=�F�HC��@=p��   ��ff��C���                                    Bxk��   T          @xQ�@:=q��������\)C���@:=q��=q��p����RC��H                                    Bxk���  
�          @p��@AG����5�/33C���@AG���  ��=q����C���                                    Bxk��L  �          @j=q@@�׿޸R�������C���@@�׿��ÿ���ȏ\C�!H                                    Bxk���  T          @g�@�R��G��p��=qC���@�R�   �   �2�C�O\                                    Bxk��  
�          @e@{�L������*�
C��@{�#�
�!��6��C���                                    Bxk��>  
�          @dz�@�J=q�"�\�7p�C��3@=L���*=q�B?�Q�                                    Bxk��  T          @b�\?��H�E��4z��U{C�B�?��H>���:�H�`Q�@��                                    Bxk��  �          @]p�?��ÿB�\�<���mp�C�{?���>L���B�\�y�H@��
                                    Bxk�%0  
�          @^�R?޸R��  ��
�,33C�s3?޸R�p���/\)�XQ�C��f                                    Bxk�3�  T          @Y��?У���
��{��\)C��{?У׿ٙ��  �.�HC���                                    Bxk�B|  �          @N{@%����H��
=��\)C�AH@%����\������=qC�1�                                    Bxk�Q"  �          @7�@{��녽L�Ϳ���C�aH@{��ff���  C�"�                                    Bxk�_�  T          @Z=q@8�ÿ�p����
=C���@8�ÿУ׿���HC�Ff                                    Bxk�nn  T          @G�@�þk���Q���C�L�@��>\��z��Q�A��                                    Bxk�}  �          @X��@�
?Y��� ���A��A�Q�@�
?�=q�Q��ffB��                                    Bxk���  �          @`��?�p�?Y���.�R�N��A���?�p�?�33���(�\B33                                    Bxk��`  T          @U@(Q�?�Ϳ�  �ffA=�@(Q�?����  �޸RA�(�                                    Bxk��  �          @Tz�@+�=�?���A�33@!G�@+���p�?��A�z�C�3                                    Bxk���  
�          @a�@Vff>���?��A#
=@���@Vff>�?.{A6=q@��                                    Bxk��R  T          @XQ�@J=q>�{�L���bff@��
@J=q?녿(���9�A"�R                                    Bxk���  �          @E�@B�\�Ǯ<��
>��RC�Z�@B�\�\���Ϳ�z�C�z�                                    Bxk��  �          @[�@Y��    ��  ���R=�\)@Y��=��
�u�\)?���                                    Bxk��D  �          @^�R@XQ�>�G��&ff�+�@�\)@XQ�?�R����
=A%�                                    Bxk� �  T          @s33@i��?G��0���&�\A@z�@i��?s33��
=�ʏ\Ai�                                    Bxk��  �          @l(�@e?#�
����G�A!@e?B�\��=q���A>=q                                    Bxk�6  "          @o\)@e�?^�R���
=A[33@e�?}p��u�o\)Aw�                                    Bxk�,�  "          @z=q@n{?��׾\��=qA�p�@n{?�Q켣�
����A�ff                                    Bxk�;�  �          @~{@s33?s33�����  A`  @s33?��
��G�����Ar{                                    Bxk�J(  "          @���@W
=@=q>W
=@8Q�B�@W
=@\)?k�AM��B�                                    Bxk�X�  T          @~�R@S33?�
=>�=q@��HA��H@S33?�G�?W
=AJ=qA��H                                    Bxk�gt  
�          @tz�@Z=q?�  =u?uA��@Z=q?�z�?�\@��A���                                    Bxk�v  "          @u@]p�?�(��&ff��
A�ff@]p�?��;8Q��*=qA�Q�                                    Bxk���  �          @�Q�@��?�(�>���@�Q�A�G�@��?���?&ffA�Al��                                    Bxk��f  "          @�33@�{?u?�@���AO\)@�{?@  ?Tz�A0Q�A"�R                                    Bxk��  "          @�  @xQ�?fff>��
@���AQ@xQ�?E�?�A�HA3�                                    Bxk���  �          @\)@tz�?�{>W
=@EA�\)@tz�?}p�?�@�{Aip�                                    Bxk��X  T          @���@~{>�G�>���@�Q�@��@~{>��
>�ff@��@�(�                                    Bxk���  �          @��\@y��?B�\>u@Z�HA/�
@y��?&ff>�ff@ϮAQ�                                    Bxk�ܤ  T          @\)@u�>�z�?s33A^ff@��
@u�    ?}p�Ah��C�                                      Bxk��J  T          @�=q@hQ���@
�HA�\)C��f@hQ쿃�
?���Aڣ�C��                                    Bxk���  
�          @��
@[��0��@%B��C�U�@[���@G�A��C�˅                                    Bxk��  T          @��H@dz�k�@(�B	33C�/\@dz�n{@G�A��\C���                                    Bxk�<  T          @��@R�\�E�@��B�C�l�@R�\��
=@�
A��C�=q                                    Bxk�%�  
�          @r�\@e?�ff?�@���A���@e?\(�?L��AD��AV�H                                    Bxk�4�  
�          @qG�@XQ�?fff?��
A��An=q@XQ�>��H?��RA�ffA33                                    Bxk�C.  �          @u�@`��?��H�L���@��A���@`��?���>k�@`��A�{                                    Bxk�Q�  T          @q�@G
=@<#�
=���B��@G
=@   ?(�AffB�H                                    Bxk�`z  T          @xQ�@:�H@\)���
���B"(�@:�H@�R>��@���B!�\                                    Bxk�o   �          @vff@+�@"�\���
�z�HB-@+�@.�R����w�B6{                                    Bxk�}�  
�          @u@��@9���333�,  BPQ�@��@>�R>.{@%�BSG�                                    Bxk��l  
Z          @u@
=q@9����p���(�BUff@
=q@HQ쾮{��(�B]�R                                    Bxk��  �          @x��@z�@p���  ���HB:�@z�@7
=�xQ��k�BK��                                    Bxk���  �          @x��@^�R?��
��ff�|  A�33@^�R?\�#�
��
A��                                    Bxk��^  �          @p  @XQ�?�=q�.{�)��A��
@XQ�?Ǯ>�z�@�z�A�                                      Bxk��  
�          @y��@`��?�G�������\)A�(�@`��?\�0���$(�A��                                    Bxk�ժ  T          @���@p��?Q녿�  ��AE�@p��?�녿u�\(�A�G�                                    Bxk��P  "          @�Q�@w
=?+��5�$  A��@w
=?Y�����H��G�AEp�                                    Bxk���  �          @���@|��?333���H��ffA ��@|��?O\)��=q�u�A9G�                                    Bxk��  T          @�  @\)>���W
=�?\)@{@\)>W
=�#�
�\)@>�R                                    Bxk�B  "          @~{@y��?�>��
@��\@�33@y��>��>�ff@�G�@�                                      Bxk��  �          @xQ�@j�H?�\)��\)����A�\)@j�H?��>�  @j�HA��\                                    Bxk�-�  �          @r�\@j�H?��>�(�@�(�A@j�H>�?
=A�
@�                                    Bxk�<4  "          @s�
@i����?��\AyG�C��q@i�����
?xQ�Al��C��H                                    Bxk�J�  
�          @c�
@Vff���?h��As�C���@Vff����?Tz�A]�C��R                                    Bxk�Y�  
(          @`  @W
=��?L��AT��C���@W
=�8Q�?!G�A%�C���                                    Bxk�h&  "          @s33@i����R?Q�AG�
C�&f@i���Tz�?(�A��C���                                    Bxk�v�  �          @g�@Z�H�@  ?W
=AX(�C��R@Z�H�s33?��AG�C�9�                                    Bxk��r  T          @o\)@,(��}p�@�\B�RC��\@,(��˅?�
=A�\)C���                                    Bxk��  
�          @mp�@8Q�k�@�B
33C��@8Q쿼(�?޸RA�\C�}q                                    Bxk���  "          @c�
@�R��G�@ ��B6�C�� @�R��@Q�B�C��f                                    Bxk��d  "          @]p�@0��?Tz�?��A�=qA��@0��>�p�?ٙ�A��@�33                                    Bxk��
  �          @`��@<(����?���A���C��@<(���\)?h��Ar{C���                                    Bxk�ΰ  �          @n{@Mp���G�?���A���C�Ff@Mp���=q?s33Am�C���                                    Bxk��V  "          @\)@W��޸R?�
=A�Q�C�` @W����R?&ffA�
C���                                    Bxk���  �          @��H@b�\���R?��A�p�C��H@b�\��?n{AR=qC�t{                                    Bxk���  T          @�Q�@[��n{?�Aڣ�C�o\@[���33?�G�A�p�C��=                                    Bxk�	H  �          @j=q@�
>.{@0  BG��@�@�
���@,(�BAC���                                    Bxk��  �          @n�R?Ǯ?!G�@Mp�By(�A���?Ǯ��\)@P  B�  C��{                                    Bxk�&�  "          @w
=?@  ?(�@hQ�B��B\)?@  ���@j=qB��C���                                    Bxk�5:  �          @y��>��<�@s�
B�  @|(�>�׿�G�@j�HB�=qC�^�                                    Bxk�C�  "          @�Q�?+��W
=@z�HB���C�:�?+����
@mp�B��=C��                                    Bxk�R�  �          @��R?#�
�B�\@�z�B��3C���?#�
���@{�B�B�C��q                                    Bxk�a,  T          @�(�?c�
���@���B��C�(�?c�
���@�=qB�u�C��                                    Bxk�o�  T          @��R?���\@�33B��)C���?���{@s�
B�#�C��=                                    Bxk�~x  T          @B�\?��@   ?�  B�Bq�\?��?\?��RB9{BT                                    Bxk��  �          @Vff?Ǯ?Ǯ@=qB>  B4��?Ǯ?Y��@/\)Bd\)A��
                                    Bxk���  �          @�?��?E�?:�HA�\)A��\?��?��?h��A���A��
                                    Bxk��j  T          @�@z�L��=#�
?�=qC�Q�@z�u<��
?+�C�33                                    Bxk��  �          @333@1G��#�
>ǮA ��C��=@1G����>�p�@�  C�w
                                    Bxk�Ƕ  �          @G�@C33���?\)A'
=C��f@C33���>��A�C�4{                                    Bxk��\  �          @.{@ �׿z�?L��A��C�t{@ �׿G�?(�AQp�C�g�                                    Bxk��  �          @,(�@�#�
?�=qA��\C�O\@�h��?^�RA��C�]q                                    Bxk��  �          @�p�@r�\<#�
?��A��>\)@r�\��(�?˅A��\C��                                     Bxk�N  
�          @�  @^�R?�?���A�(�A��@^�R?��H?�z�A֣�A�Q�                                    Bxk��  �          @�=q@b�\�.{?�ffA��C��)@b�\�&ff?�Q�A��HC��{                                    Bxk��  �          @�G�@Y��>��@%�B�@\)@Y�����@!�BG�C�n                                    Bxk�.@  �          @��\@c33��z�@�B	33C��f@c33�n{@��A�\)C���                                    Bxk�<�  �          @��\@]p�>L��@!�B��@Vff@]p���@   B(�C��                                    Bxk�K�  �          @���@Vff��
=@	��A��C�l�@Vff����?�Q�A�=qC��                                    Bxk�Z2  �          @�Q�@\�Ϳ�  ?�\A�(�C��@\���
=q?��RA�G�C���                                    Bxk�h�  �          @�\)@l(����?˅A�\)C�5�@l(���Q�?�Q�A��HC���                                    Bxk�w~  
�          @�  @`  ��(�?ǮA��C��@`  �z�?�ffAf�\C��\                                    Bxk��$  "          @�
=@l�Ϳ�=q?��
A��C�ff@l�Ϳ�\)?Q�A3
=C���                                    Bxk���  �          @���@W��G�?�{A�C��@W��#33?5A�C�|)                                    Bxk��p  �          @���@Vff�  ?�A��HC�3@Vff�"�\?G�A'\)C�h�                                    Bxk��  
�          @�Q�@R�\�ff?�z�A�(�C��R@R�\�p�?�ffAd��C��
                                    Bxk���  T          @���@Z�H��(�?��A��RC��@Z�H��
?xQ�AS�
C��q                                    Bxk��b  �          @�ff@u����?J=qA.ffC���@u��{>\@�  C��f                                    Bxk��  T          @�@h�ÿ˅?��A�  C�0�@h�ÿ�\)?Tz�A7\)C�aH                                    Bxk��  T          @�ff@aG���{?�=qA�33C�l�@aG����?�z�A��C��H                                    Bxk��T  T          @��@e��˅?�G�A�
=C��@e���
=?�ffAf�RC���                                    Bxk�	�  �          @�Q�@tzῸQ�?���A���C��3@tz�ٙ�?G�A)G�C�                                    Bxk��  
�          @�ff@�  �Q�?n{ALQ�C�,�@�  ���?.{Ap�C��)                                    Bxk�'F  T          @�p�@|�;��?�(�A�=qC��
@|�ͿE�?�ffAj�RC�w
                                    Bxk�5�  
�          @�@aG��k�?��HA�Q�C���@aG����?�33A��RC�1�                                    Bxk�D�  �          @�z�@^�R�\)@��A���C��H@^�R�0��@�A�\C�l�                                    Bxk�S8  
�          @�p�@g
=>W
=@33A�  @XQ�@g
=���
@�A�(�C�q�                                    Bxk�a�  
�          @�  @s33��?�{A��C�@s33��\?��Aȏ\C�(�                                    Bxk�p�  �          @�Q�@I���}p�@)��B
=C�B�@I�����@�
B��C�<)                                    Bxk�*  T          @�  @E��G�@'�BQ�C��3@E���@��A�z�C�O\                                    Bxk���  �          @��@8�ÿ�
=@6ffB)z�C��@8�ÿ�\)@��B�RC���                                    Bxk��v  �          @�\)@:=q��=q@7�B*��C�Ф@:=q���
@\)B�RC�E                                    Bxk��  �          @�33@`  �
=@33A�C�:�@`  ���?�=qAӮC�T{                                    Bxk���  
�          @�z�@n�R�O\)?�=qA��C��
@n�R��Q�?���A��RC�%                                    Bxk��h  T          @�ff@\)���\@C33B>  C��=@\)�   @(Q�B�\C���                                    Bxk��  �          @��
@B�\�У�@{B=qC��@B�\���?�p�A��
C�e                                    Bxk��  �          @��\@C�
�z�?�ffA��C���@C�
��?:�HA,z�C�T{                                    Bxk��Z  T          @�(�@�
���H@QG�BQ=qC�P�@�
�\)@2�\B)�HC�B�                                    Bxk�   
�          @���@Q�z�H@Mp�BMG�C�� @Q��\@7
=B/�
C���                                    Bxk��  
�          @��?��=���@l��B��q@u?���W
=@g
=B�#�C��\                                    Bxk� L  �          @�z�?�
==�\)@vffB�33@.�R?�
=�h��@o\)B�=qC�                                    Bxk�.�  T          @��?��R�Ǯ@y��B�Q�C�S3?��R��{@k�B}C��                                    Bxk�=�  �          @��?aG�?�@z�HB�A�  ?aG���@z�HB�ffC��R                                    Bxk�L>  �          @�(�?�\)?�\)@b�\Bu�B4Q�?�\)>�(�@q�B�u�A��                                    Bxk�Z�  "          @�(�?���?�{@Z�HBa�B�?���>�@i��B{Q�Aa                                    Bxk�i�  �          @�?��R?Tz�@b�\Bj�\A�=q?��R���
@hQ�Bu33C�˅                                    Bxk�x0  "          @�{@#33=�@S33BQ{@%@#33�8Q�@N{BJG�C�!H                                    Bxk���  �          @��
@�;k�@Q�BTz�C�]q@�Ϳ��@G�BF  C�w
                                    Bxk��|  T          @���?�z�?��
@\(�Bk�\B={?�z�?(�@n{B�
=A�                                      Bxk��"  �          @�G�?�
=?��@K�BT��BG�?�
=?��@[�Bn��A~=q                                    Bxk���  �          @}p�@W��(��?�Q�A�C�u�@W���\)?��HA�C��f                                    Bxk��n  
�          @�  @Dz�&ff@{B�C�\@Dz῞�R@�RB�
C���                                    Bxk��  "          @~{@A논�@��B��C��R@A녿!G�@33B�RC�+�                                    Bxk�޺  T          @~�R@I����G�@Q�B(�C��@I���u?���A�33C���                                    Bxk��`  "          @~{@K��!G�?�G�A�C�^�@K���ff?��A��
C��)                                    Bxk��  �          @�Q�@G�?���?�Q�A�A��R@G�?Y��@�RB
33At(�                                    Bxk�
�  "          @�G�@qG�?   ?���A��@�=q@qG�>W
=?�Q�A�ff@L��                                    Bxk�R  
�          @��H@i����?��HA�\)C��{@i�����?У�A�z�C���                                    Bxk�'�  
�          @��\@Vff��ff@��B  C�1�@Vff�z�H@G�A�  C��)                                    Bxk�6�  T          @���@\(��fff?���A�G�C��f@\(���=q?��A�C�u�                                    Bxk�ED  
�          @y��@O\)>�Q�@	��BQ�@�z�@O\)�8Q�@
�HB��C�k�                                    Bxk�S�  �          @w�?�@�@/\)B6Q�Bh?�?��@N{Bd33B=33                                    Bxk�b�  �          @vff@z�?��H@"�\B'
=B-��@z�?��\@<��BJ(�A�z�                                    Bxk�q6  
�          @w�@33?��
@(Q�B0=qB$(�@33?�=q@>�RBP�HA�Q�                                    Bxk��  �          @u@#�
?��@��Bp�A�p�@#�
?(��@(��B3�Af=q                                    Bxk���  �          @r�\@e>�  ?�=qA�(�@}p�@e��?�{A�C�˅                                    Bxk��(  �          @\)?��@
=@HQ�BS��B}�?��?��
@c�
B��fBH�H                                    Bxk���  �          @|(�@?\)?��\@
=qB	z�A�33@?\)>�@B33AQ�                                    Bxk��t  �          @���@u=���?�A��?�  @u�B�\?�z�A���C��{                                    Bxk��  �          @�G�@B�\?fff?�
=A���A�(�@B�\>���@B	\)@�                                    Bxk���  �          @���@'
=?���@5B.
=A�R@'
=?5@G
=BC�At��                                    Bxk��f  �          @�p�@'�?�z�@:=qB1Q�A�z�@'�?(��@J�HBF
=AaG�                                    Bxk��  �          @�@#33?�@8Q�B2p�A�=q@#33?+�@H��BGAk33                                    Bxk��  "          @�ff@2�\?�G�@%BA�
=@2�\?Tz�@8Q�B2��A��\                                    Bxk�X  �          @��@>{@#�
?��\A�\)B#�@>{@
�H?���A��B�
                                    Bxk� �  
�          @�@>�R?�@��B33A�p�@>�R?�p�@(��B{A��                                    Bxk�/�  "          @�{@U�?\@ ��A�Q�A�(�@U�?z�H@z�B
=A���                                    Bxk�>J  T          @�
=@e?���?��A�z�A��R@e?(�@A���A33                                    Bxk�L�  �          @�Q�@}p�>\?�G�A���@���@}p�    ?ǮA�(�=L��                                    Bxk�[�            @���@w
=>���?��A��@�\)@w
=�\)?�A�p�C��                                    Bxk�j<  �          @�  @|�;L��?�  A�ffC���@|�Ϳ\)?�z�A���C��R                                    Bxk�x�  
�          @�  @��׾8Q�?��A��C��q@��׿   ?�  A��C�|)                                    Bxk���  y          @�z�@���=�G�?�(�A��?�  @��׾��
?ٙ�A��RC��H                                    Bxk��.  �          @��
@~{���H?޸RA�Q�C�|)@~{�fff?���A���C���                                    Bxk���  �          @�z�@}p��(�?�G�A�{C�� @}p����\?ǮA�p�C�˅                                    Bxk��z  T          @��
@vff�xQ�?��
A���C���@vff��\)?��RA�=qC�/\                                    Bxk��   T          @�G�@l(����R?��
A��
C��@l(�����?�\)At(�C��=                                    Bxk���  �          @��@|�Ϳ&ff?ǮA�(�C�T{@|�Ϳ�G�?�{A�33C���                                    Bxk��l  �          @�  @q녿�?��A�\)C��@q녿n{?�\)A�Q�C�3                                    Bxk��  �          @���@l(���Q�@��A�RC�N@l(����@�
A�Q�C�^�                                    Bxk���  "          @�G�@c�
�W
=@�
Bz�C�O\@c�
�B�\@(�A��C��)                                    Bxk�^  T          @���@�33�.{?��
A_�C�S3@�33�fff?W
=A5�C���                                    Bxk�  
�          @�Q�@�z���?W
=A7\)C��@�z�(��?5AG�C�|)                                    Bxk�(�  �          @��@~{�n{?�G�A^�RC�q�@~{���?B�\A&�\C��)                                    Bxk�7P  �          @���@s33��{?�Q�A��C�l�@s33�L��?�ffA�p�C��                                    Bxk�E�  T          @�=q@dz�W
=@�RA���C�L�@dz�=p�@
=A��HC�#�                                    Bxk�T�  �          @�=q@e��#�
@��BQ�C���@e��(�@z�BG�C�!H                                    Bxk�cB  �          @���@c33�u@ffBG�C���@c33��R@G�B ��C�f                                    Bxk�q�  
�          @���@Z=q>��R@!�B��@��@Z=q���R@!�B��C�n                                    Bxk���  /          @�G�@[�?��@!G�B�HA�H@[��u@%�B�
C�u�                                    Bxk��4  y          @���@c�
?��@�
B�A{@c�
=#�
@Q�B{?z�                                    Bxk���  �          @���@q�>�  ?�p�A�
=@mp�@q녾u?��RA�33C�8R                                    Bxk���  T          @��@w
=�#�
?�  A���C��3@w
=��
=?ٙ�A���C�޸                                    Bxk��&  �          @�  @�  �\)?�ffA��HC�
=@�  �Y��?���Aw\)C���                                    Bxk���  T          @�  @|(���Q�?(�A�C��
@|(����>�  @Z�HC�L�                                    Bxk��r  T          @���@��ÿ���>Ǯ@�  C���@��ÿ�
==L��?.{C�33                                    Bxk��  �          @���@\)����?:�HA�C�t{@\)�\>Ǯ@��C���                                    Bxk���  �          @�  @}p���z�?z�@��\C�:�@}p���  >k�@Dz�C��)                                    Bxk�d  �          @��R@~�R���>�33@���C���@~�R����<�>���C��q                                    Bxk�
  �          @��R@�  ��(�>�@ǮC��f@�  ���>��@
=C��                                    Bxk�!�  �          @�{@}p����\>�(�@�p�C�
@}p����=�?���C���                                    Bxk�0V  /          @�p�@}p���  �.{�z�C�8R@}p���
=�����
C��R                                    Bxk�>�  y          @���@��׿s33������C�T{@��׿k���  �aG�C��\                                    Bxk�M�  T          @��H@~�R��Ϳ#�
�z�C��@~�R�Ǯ�@  �*ffC�8R                                    Bxk�\H  "          @��@z�H��G��&ff���C���@z�H�Tz�^�R�C�C��                                    Bxk�j�  T          @��@xQ�h�ÿxQ��[�
C�n@xQ�&ff��z����C�@                                     Bxk�y�  T          @��H@s�
�5�������C���@s�
��녿�����C��\                                    Bxk��:  "          @���@r�\�J=q�n{�X��C��@r�\��Ϳ���~�HC��                                    Bxk���  T          @���@k����
��ff�qC�ff@k���  ��=q��\)C�l�                                    Bxk���  T          @���@p  ��  ��R�ffC��q@p  ���ÿfff�O�C��                                    Bxk��,  �          @���@tz�}p��aG��P��C���@tz�h�þ�ff��=qC�L�                                    Bxk���  T          @�Q�@u�xQ�z���C��\@u�L�ͿJ=q�6=qC�R                                    Bxk��x  �          @���@u�E��Q��<  C�T{@u�\)�xQ��ap�C��H                                    Bxk��  �          @���@n�R�Tzῑ���G�C��
@n�R�
=q�����
=C���                                    Bxk���  �          @}p�@u?!G������A�@u?=p���p���z�A.�H                                    Bxk��j  �          @\)@tz��ff�=p��-�C��H@tzᾊ=q�Q��B=qC��                                    Bxk�  �          @\)@w
=��
=�Y���E�C��@w
=�W
=�k��V�HC�t{                                    Bxk��  "          @}p�@(���z�\(��f�RC���@(�ÿ���=q��  C��                                    Bxk�)\  �          @�  ?���u��\)����C���?���j�H��
=��G�C�\                                    Bxk�8  
�          @�  ?�ff�n{�   ��RC���?�ff�`  ��{��=qC�*=                                    Bxk�F�  "          @�  ?�G��k��:�H�*{C�y�?�G��Z=q��=q����C�.                                    Bxk�UN  T          @\)?��R�k��5�&�\C�K�?��R�Z=q�Ǯ��{C���                                    Bxk�c�  S          @~{?У��`  �J=q�8��C���?У��N{�˅��=qC�xR                                    Bxk�r�  T          @}p�?ٙ��_\)�333�$(�C��3?ٙ��N�R��  ��G�C���                                    Bxk��@  �          @s�
@�
�>{�\)�ffC��=@�
�7
=�Tz��LQ�C�xR                                    Bxk���  �          @u@��;������C�t{@��-p����\��G�C���                                    Bxk���  
�          @u@@�׿@  ��\)��z�C��@@�׾�z���R��C�AH                                    Bxk��2  T          @y��@���1녿�  ���C�h�@���=q��{��C�t{                                    Bxk���  �          @z�H@\)��ÿ�ff��(�C��@\)��33��
���C�Q�                                    Bxk��~  �          @{�@Fff������z�C��)@Fff���
��ff��(�C��                                    Bxk��$  �          @{�@>�R��R����C�"�@>�R�Q�5�(Q�C��
                                    Bxk���  �          @w�@"�\�8Q�#�
��C��3@"�\�2�\�5�*�\C�%                                    Bxk��p  T          @vff@0  �)��>�z�@��C��@0  �(�þ�{���
C��                                    Bxk�  T          @w
=@�R�(Q�?��\A�z�C���@�R�6ff?�RA�
C��f                                    Bxk��  �          @vff?��\�/\)@
=qB=qC�k�?��\�J�H?���A��RC��                                    Bxk�"b  �          @x��?����1G�?�ffA��C���?����G
=?���A��C�*=                                    Bxk�1  �          @u�?���5�@p�B��C��?���QG�?��RA�G�C�ٚ                                    Bxk�?�  �          @tz�?޸R�&ff@Q�B	(�C��H?޸R�AG�?��HA��C��                                    Bxk�NT  �          @u@4z��Q�?��RA���C�l�@4z���H?s33Adz�C��                                    Bxk�\�  �          @�  @@  ��ff?�A��
C��H@@  ��?�\)A�C���                                    Bxk�k�  �          @}p�@7
=���@G�A�p�C�H@7
=�p�?�ffA�33C�&f                                    Bxk�zF  	�          @|(�@+��#33?xQ�Ak33C�1�@+��-p�>�33@�C�XR                                    Bxk���  T          @xQ�?��R�S33�����HC��=?��R�K��fff�XQ�C�f                                    Bxk���  "          @z=q?�Q��_\)��=q����C��?�Q��U���=q��p�C�q�                                    Bxk��8  �          @{�?����\�;�=q�z=qC��?����R�\�����}p�C�|)                                    Bxk���  T          @z�H?У��aG���Q���G�C�h�?У��U�����Q�C�                                      Bxk�Ä  �          @w�@>�R��H=u?c�
C�t{@>�R����\���
C��                                    Bxk��*  �          @w�@N{�
=>B�\@8��C�e@N{�ff���R��G�C�y�                                    Bxk���  T          @xQ�@P�׿������
��z�C��@P�׿�녾���{C��\                                    Bxk��v  �          @y��@U���p��\)�Q�C��f@U���녿���33C�9�                                    Bxk��  �          @x��@4z�� �׿����C�4{@4z���\����
=C�q�                                    Bxk��  T          @vff@;������{��{C�^�@;��  �fff�Z=qC�AH                                    Bxk�h  
�          @tz�@<����H���Ϳ�
=C�W
@<�����(���33C�޸                                    Bxk�*  T          @u@Z=q�У׾�����G�C�7
@Z=q��G��5�+�C��                                    Bxk�8�  �          @w
=@N�R���R������ffC�7
@N�R���ͿJ=q�>ffC�{                                    Bxk�GZ  �          @u@3�
�Q쿰�����RC�s3@3�
��  ����Q�C�R                                    Bxk�V   �          @vff@:=q��H��ff����C�q@:=q�  ��  �t��C�&f                                    Bxk�d�  �          @xQ�@W
=�޸R�����C�G�@W
=���ÿp���d  C�~�                                    Bxk�sL  �          @s33@^�R��{���
�{\)C�*=@^�R�W
=��G���
=C�>�                                    Bxk���  T          @q�?��H�O\)�=p��5�C���?��H�>{��(���(�C��=                                    Bxk���  �          @k�?��H�p���33��\)C���?��H��G����\)C��                                    Bxk��>  x          @h��@2�\��
=��(����C��\@2�\�=p���Q��  C���                                    Bxk���  T          @n�R@{�=q��z���\)C���@{� �׿�
=��  C�e                                    Bxk���  �          @p  @%��׿��R���C�l�@%���Ϳ�p�����C�9�                                    Bxk��0  �          @n�R?����;������  C�'�?����%���ff��=qC��R                                    Bxk���  �          @mp�@����\�����{C�%@���������
C���                                    Bxk��|  �          @qG�@"�\�G�������\C��@"�\��33�����C���                                    Bxk��"  �          @mp�@�R�%�s33�m�C��H@�R��\���
����C��q                                    Bxk��  �          @k�@(��(Q�B�\�?�C�e@(��Q쿮{���C��
                                    Bxk�n  T          @p��@G��;���\���C��R@G��.�R��Q���  C��R                                    Bxk�#  �          @vff@{�G��8Q��)��C��{@{�?\)�h���Z�HC�H�                                    Bxk�1�  �          @s33@33�G
=����\C���@33�9����  ��p�C���                                    Bxk�@`  �          @h��?��
�8�ÿ��
��G�C���?��
� �׿�
=� �C��{                                    Bxk�O  �          @j=q?��H�C�
��G���=qC��
?��H�/\)�ٙ���(�C��q                                    Bxk�]�  �          @h��?�(��7�������C�q�?�(��p��33�	��C�z�                                    Bxk�lR  T          @g�?���.�R��(�����C�=q?����
�z��G�C���                                    Bxk�z�  �          @p  ?�(��8Q�������C��?�(��
=�{�&�\C��                                    Bxk���  �          @qG�?�Q���R�=q� Q�C�R?�Q���;��M��C��                                    Bxk��D  �          @g�@
=���ff�홚C���@
=�����33�\)C�T{                                    Bxk���  �          @i��?ٙ��z����(C��q?ٙ���Q��6ff�O�
C��                                    Bxk���  �          @s�
@W
=��p�?z�HAp  C�#�@W
=��?(�A
=C��\                                    Bxk��6  �          @s�
@^{��33?&ffAp�C�f@^{��G�>���@���C�9�                                    Bxk���  �          @tz�@^{��  >��R@��RC�S3@^{���
�L�ͿG�C�)                                    Bxk��  �          @u�@_\)��G�>.{@$z�C�O\@_\)��G��B�\�8Q�C�S3                                    Bxk��(  �          @s33@p�׾�>#�
@��C�` @p�׿   =#�
?�RC�1�                                    Bxk���  �          @mp�@j�H����>��
@�{C���@j�H��>aG�@UC�j=                                    Bxk�t  T          @mp�@k��8Q�>W
=@L��C��
@k��aG�>#�
@(�C�AH                                    Bxk�  "          @o\)@n{�#�
>.{@#33C���@n{�B�\=�?��C��                                     Bxk�*�  �          @p��@n{�Ǯ��{���C��@n{������
=��  C��R                                    Bxk�9f  T          @r�\@j=q�@  �\)��
C�1�@j=q����8Q��/�
C�Z�                                    Bxk�H  T          @n�R@h�ÿ333�L���G
=C���@h�ÿ#�
��Q����C��                                    Bxk�V�  �          @h��@^�R�z�J=q�J�HC�J=@^�R��p��h���h��C���                                    Bxk�eX  �          @g
=@S�
���\�n{�q��C�g�@S�
�E���33���C�n                                    Bxk�s�  T          @e�@N�R��p��h���k33C��R@N�R�z�H����G�C��
                                    Bxk���  �          @hQ�@^{�fff�Ǯ��Q�C���@^{�G�����z�C��                                     Bxk��J  �          @g�@e��   ��\)���C�@e���׾8Q��8Q�C�J=                                    Bxk���  �          @i��@\�Ϳ(��?Tz�AT(�C��3@\�ͿW
=?&ffA%�C�*=                                    Bxk���  �          @l(�@C�
���?�z�A�z�C��@C�
�z�H?�(�A�C��                                    Bxk��<  �          @n{@B�\�(��@33B\)C��@B�\����?�=qA�C���                                    Bxk���  �          @k�@U���?��A��C�L�@U�^�R?�(�A�{C��3                                    Bxk�ڈ  �          @l��@S�
�!G�?�G�A�Q�C���@S�
�z�H?��A�(�C���                                    Bxk��.  "          @l��@I���@  ?�\A�p�C�P�@I����33?��A�p�C��R                                    Bxk���  �          @l(�@A녿u?�{A���C�4{@A녿���?�=qA���C��f                                    Bxk�z  
�          @h��@C�
�aG�?�  A�ffC��3@C�
���
?��RA�p�C���                                    Bxk�   �          @q�@B�\��ff?�\A�33C�k�@B�\��Q�?�z�A�=qC�}q                                    Bxk�#�  �          @s�
@,(���ff@
=qBffC��@,(��G�?�p�A�p�C���                                    Bxk�2l  T          @qG�@8�ÿ�G�@�B
=C�@ @8�ÿ��H?�A�\)C��R                                    Bxk�A  �          @n�R@#33�Ǯ@p�BC�=q@#33�33?�\A�33C��q                                    Bxk�O�  �          @h��@>{�Y��?��A��C�H@>{���
?�\)AՅC�Y�                                    Bxk�^^  �          @h��@N�R�+�?�  A��C�(�@N�R��G�?�ffA��
C�S3                                    Bxk�m  �          @mp�@W
=�8Q�?��A���C���@W
=���
?�
=A���C�xR                                    Bxk�{�  �          @l��@S�
�aG�?���A��
C���@S�
��?���A���C�>�                                    Bxk��P  T          @j=q@Mp���33?�ffA��C�#�@Mp���
=?}p�A{�C��)                                    Bxk���  �          @k�@J=q��\)?��RA�Q�C�<)@J=q�У�?^�RA\Q�C�S3                                    Bxk���  T          @mp�@S33��?ǮAǙ�C��@S33�n{?�\)A���C�%                                    Bxk��B  �          @l(�@U�aG�?��\A�C��=@U��z�?��\A��C�u�                                    Bxk���  �          @n{@W����\?�z�A��
C��R@W����\?aG�A[�C���                                    Bxk�ӎ  �          @n�R@Z=q�h��?���A���C���@Z=q��?s33Al  C���                                    Bxk��4  �          @mp�@R�\�5?�z�A���C��=@R�\���
?�Q�A��HC�P�                                    Bxk���  "          @g�@P  ����?��A��
C���@P  �B�\?�33A�(�C�l�                                    Bxk���  �          @g�@L�;k�?�z�A�G�C��q@L�Ϳ�R?ǮA�z�C��                                     Bxk�&  �          @hQ�@G
=>�p�?�=qA�=q@��H@G
=���
?�{A��C�=q                                    Bxk��  �          @h��@S�
>aG�?�Q�A��@xQ�@S�
��?���A�p�C�ٚ                                    Bxk�+r  �          @c�
@R�\�u?�ffA��HC�u�@R�\�\?�  A�\)C���                                    Bxk�:  �          @fff@^{�8Q�?
=qA�
C�*=@^{�Tz�>�33@�p�C�K�                                    Bxk�H�  �          @c33@Tz῝p�>aG�@_\)C��{@Tz῞�R��Q쿺�HC��
                                    Bxk�Wd  �          @e�@=p���
=�   � ��C�o\@=p���  �u�x��C���                                   Bxk�f
  �          @g
=@<����\��(����
C��@<�Ϳ�\)�k��k\)C��                                   Bxk�t�  4          @j=q@AG����R�   ���\C�J=@AG����xQ��v=qC��f                                    Bxk��V  �          @g�@<���   �����C��@<�Ϳ����\��p�C�=q                                    Bxk���  �          @^�R@���G����
���\C�k�@�׿�녿�ff��p�C�                                    Bxk���  �          @b�\@
=���������C��q@
=��p��޸R��C��                                    Bxk��H  �          @Z=q@���������Q�C��{@���   ������{C��f                                    Bxk���  �          @[�@����0���9�C��@���ff���R��Q�C�p�                                    Bxk�̔  �          @\(�@33�   �����ָRC�J=@33����  ��Q�C�T{                                    Bxk��:  �          @[�@*=q�녾�ff��z�C�P�@*=q��{�n{���\C��f                                    Bxk���  �          @Z=q@5�����>8Q�@EC��f@5�����=q��(�C���                                    Bxk���  �          @[�@"�\�G�=�Q�?���C��@"�\�{��� z�C�n                                    Bxk�,  �          @[�@#33�G��#�
���C�(�@#33��Ϳ��p�C���                                    Bxk��  �          @Z=q@.{�   ������C��
@.{��\)�=p��J�HC��R                                    Bxk�$x  �          @`��@7���p����R���C��{@7����J=q�Pz�C���                                    Bxk�3  �          @^{@z���ÿ�z���{C��H@z��p������HC���                                    Bxk�A�  �          @[�@   �	���h���w
=C���@   ���Ϳ�z��ÅC��R                                    Bxk�Pj  �          @^�R@4z���ÿB�\�LQ�C���@4z��=q��Q����C�ff                                    Bxk�_  �          @b�\@L�Ϳ�  ��Q쿾�RC�g�@L�Ϳ�
=�����C��                                    Bxk�m�  �          @fff@P  ���?s33Au��C�S3@P  ���?!G�A#
=C�˅                                    Bxk�|\  �          @g�@`  �Ǯ?:�HA:ffC��@`  �\)?��A33C�s3                                    Bxk��  �          @i��@Z�H���?�\A   C��q@Z�H��p�>L��@N�RC��                                    Bxk���  �          @l��@]p���  >��
@�\)C��@]p����    ���
C��                                    Bxk��N  �          @n{@a녿��>u@n{C�@a녿�z�L�ͿJ=qC��H                                    Bxk���  �          @l��@c33�}p�=#�
?\)C�:�@c33�xQ�W
=�S33C�e                                    Bxk�Ś  �          @k�@^�R����=u?uC��)@^�R��{�aG��aG�C�#�                                    Bxk��@  �          @g�@`  �aG�=��
?�  C��=@`  �^�R�����
C���                                    Bxk���  �          @g
=@`�׿+���\)��  C��3@`�׿z��G����C�N                                    Bxk��  �          @i��@a녿c�
�.{�)��C��@a녿Q녾Ǯ��33C�s3                                    Bxk� 2  �          @k�@g����R��p����C��
@g��L�;�(���Q�C�c�                                    Bxk��  �          @j=q@Z�H��������C��
@Z�H�������33C�aH                                    Bxk�~  �          @aG�@,����R�u�p��C�9�@,���	����R�"�RC�Ǯ                                    Bxk�,$  
�          @^{@>�R��  <#�
=���C���@>�R��Q��G���  C�1�                                    Bxk�:�  �          @fff@TzῨ�þu�s33C�"�@Tz῜(��\)���C���                                    Bxk�Ip  �          @fff@O\)��G����
����C��H@O\)���׿0���0��C�u�                                    Bxk�X  �          @e�@R�\���;����ffC��\@R�\���R�����C��                                    Bxk�f�  �          @c�
@L(������  ��  C�q@L(���
=�!G��#�C��3                                    Bxk�ub  �          @[�@;��ٙ������{C��\@;���=q�0���:�\C�ٚ                                    Bxk  �          @W�@0  ��=q>�ff@��C�33@0  ��׼����C��                                    Bxk�  �          @S�
@6ff�У�>���@��C��@6ff��z���
=C��                                    Bxk¡T  �          @U�@1G���G�?��AQ�C��f@1G���=��
?�C�.                                    Bxk¯�  T          @b�\@7���\>���@��C�S3@7��33�k��mp�C�B�                                    Bxk¾�  T          @\(�@1�� ��>B�\@N{C�3@1녿��R������\)C�1�                                    Bxk��F  T          @\(�@3�
��(�>#�
@&ffC�xR@3�
��Q쾳33��C��                                    Bxk���  T          @Y��@:=q��>��
@�  C��@:=q�ٙ����   C�ٚ                                    Bxk��  
�          @XQ�@2�\����>�{@�\)C�=q@2�\��\)����$z�C��                                    Bxk��8  �          @Z�H@7���ff>��
@�C��{@7���=q����#�
C�˅                                    Bxk��  �          @c33@B�\��=��
?��
C��H@B�\��\�Ǯ�ʏ\C��=                                    Bxk��  �          @g
=@N�R�˅=�Q�?�C��@N�R�Ǯ���
���C�                                      Bxk�%*  �          @fff@P  ��
=>�z�@�\)C�"�@P  ������Q쿰��C��\                                    Bxk�3�  �          @dz�@6ff�G����C�S3@6ff���&ff�*�\C�                                    Bxk�Bv  T          @b�\@-p��\)�����p�C�9�@-p��ff�Tz��YG�C�%                                    Bxk�Q  �          @e@L(��˅>�z�@��RC��R@L(���{���C���                                    Bxk�_�  �          @e@L�Ϳ��>��@�\)C�.@L�Ϳ���=L��?O\)C���                                    Bxk�nh  �          @g�@G
=���=�Q�?��RC�\@G
=��  �\��  C�Q�                                    Bxk�}  �          @h��@=p����z���G�C�^�@=p���Q�Q��Qp�C�XR                                    BxkË�  �          @h��@'
=�=q����(�C���@'
=����z���=qC��                                    BxkÚZ  �          @hQ�@(���p�������z�C��@(�ÿ��Ϳ�{�ԣ�C�y�                                    Bxké   �          @e@   �G���G���C���@   ��ff���R�G�C�\                                    Bxk÷�  �          @aG�?�{�   ��p����C�Y�?�{�G�����C�P�                                    Bxk��L  �          @c33?��R� ��?��RA��C�33?��R�4z�?G�AJ�RC���                                    Bxk���  �          @a�@�H��
?�RA((�C�,�@�H��ü#�
�\)C��                                    Bxk��  �          @e�@z��9��=�?�z�C��)@z��5��&ff�(  C��                                    Bxk��>  T          @b�\@#33��=�G�?��HC�1�@#33���
=q���C���                                    Bxk� �  T          @e@,(��>�{@�
=C��H@,(����z����
C�w
                                    Bxk��  �          @c33@8Q��   ?
=A  C�� @8Q���=#�
?��C��                                    Bxk�0  �          @b�\@=p����
?�RA"{C�y�@=p����=�@   C��f                                   Bxk�,�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxk�;|  
          @i��@.�R�?(�A��C��3@.�R��H�#�
�\)C�=q                                    Bxk�J"  �          @l(�@<(��
�H>�
=@�33C��f@<(���;.{�*=qC��\                                    Bxk�X�  �          @i��@*�H�G�?�(�A�p�C�p�@*�H�?aG�A]�C�c�                                    Bxk�gn  �          @p��@7���  �ٙ���\)C�Q�@7���p��
=�G�C�ff                                    Bxk�v  �          @n{@*�H�У׿��R��C�S3@*�H���
�
=���C�j=                                    BxkĄ�  �          @n{@0�׿�׿У��ϮC��@0�׿�\)�z��\)C��                                    Bxkē`  �          @g
=@/\)�33��33��=qC���@/\)��z��33��=qC�aH                                    BxkĢ  �          @a�@333��(���{����C�:�@333��{���
���
C�3                                    Bxkİ�  �          @`��@)����p���p��ȸRC�h�@)�����\����33C�33                                    BxkĿR  �          @^�R@+����H��33��ffC���@+����\����\)C�P�                                    Bxk���  �          @_\)@.�R��녿��\��{C���@.�R��ff���R����C�>�                                    Bxk�ܞ  �          @`  @0���ff���
��Q�C�\)@0�׿�Q�aG��h��C�xR                                    Bxk��D  �          @`��@{�(���z���G�C���@{�G��p���z�RC���                                    Bxk���  �          @^{@(���Q�+��2=qC���@(�ÿ�\)��(���p�C�U�                                    Bxk��  �          @^{@*�H���+��0��C�� @*�H��{��(���(�C���                                    Bxk�6  �          @aG�@*=q���.{�3\)C�P�@*=q����  ��
=C�!H                                    Bxk�%�  �          @a�@*�H��R����(�C��@*�H���R�������C���                                    Bxk�4�  �          @k�@0  �Q쾮{���\C��{@0  ��Ϳz�H�w�
C���                                    Bxk�C(  �          @n�R@1���L���F�HC���@1��녿�z���(�C��\                                    Bxk�Q�  �          @mp�@7
=�녿���RC��H@7
=�녿�
=��(�C�P�                                    Bxk�`t  �          @p  @:=q�z�\)���C��R@:=q�(��E��@  C���                                    Bxk�o  �          @qG�@5��(���{��C��f@5��  ��  �v{C��f                                    Bxk�}�  �          @r�\@2�\�"�\>#�
@�C��@2�\�\)�����C�)                                    BxkŌf  �          @q�@(Q��,(����R����C�'�@(Q��   �����z�C�5�                                    Bxkś  �          @p��@)���(�þ�\)��G�C���@)���p���  �w\)C��f                                    Bxkũ�  �          @r�\@   �6ff=L��?L��C���@   �0  �=p��333C�%                                    BxkŸX  �          @q�@#�
�0��>�=q@��C�e@#�
�.�R��\��  C��3                                    Bxk���  �          @s�
@;����>�@�  C�l�@;����L���AG�C�0�                                    Bxk�դ  �          @tz�@6ff��?8Q�A.�RC���@6ff�!�<��
>�\)C�.                                    Bxk��J  �          @tz�@AG���?}p�Ao�C�w
@AG���
>�Q�@�(�C�J=                                    Bxk���  �          @s�
@333�{?333A*�HC�P�@333�$z�#�
�8Q�C��H                                    Bxk��  �          @tz�@,���{?��A~�RC��f@,���*�H>��R@�ffC���                                    Bxk�<  �          @s33@:�H�z�?�G�A��\C�T{@:�H�?!G�A�C��f                                    Bxk��  �          @qG�@>{��{?���A�G�C��q@>{�	��?@  A8Q�C��                                    Bxk�-�  �          @mp�@AG���
=?^�RAYp�C���@AG��>�z�@�  C���                                    Bxk�<.  �          @i��@*=q��(�?��A�C�� @*=q�z�?k�Aj�HC�p�                                    Bxk�J�  �          @i��@0  �   ?�A��C��@0  �\)?\)AffC�c�                                    Bxk�Yz  �          @e�@:�H��=q?p��Atz�C���@:�H�G�>Ǯ@ƸRC���                                    Bxk�h   �          @hQ�@9����33?c�
AeC�j=@9���z�>��
@���C�>�                                   Bxk�v�  T          @e@>{��\)?8Q�A9C��=@>{�   >#�
@%C��                                   Bxkƅl  �          @j=q@@  ���H?=p�A;33C�n@@  �>#�
@(�C��3                                    BxkƔ  �          @j�H@HQ��{>�\)@�33C���@HQ��{������
C���                                    BxkƢ�  �          @j=q@C33�У�?�Q�A�=qC��{@C33��33?.{A+\)C�
=                                    BxkƱ^  �          @hQ�@B�\��
=?��A��C�aH@B�\��G�?aG�A`(�C���                                    Bxk��  �          @g
=@9����Q�?˅A���C�Ǯ@9����=q?�{A�G�C���                                    Bxk�Ϊ  �          @h��@@�׿�p�?�\)A��C��=@@�׿�?c�
Ac33C�~�                                    Bxk��P  �          @k�@HQ쿺�H?��\A��
C���@HQ��G�?L��AI�C�P�                                    Bxk���  T          @p��@Fff���
?��HA�p�C�޸@Fff���?uAm��C�Y�                                    Bxk���  �          @l��@@  ��Q�?�=qA���C�K�@@  �   ?J=qAE�C�"�                                    Bxk�	B  �          @k�@9����
=?�G�A��
C��{@9����\?uAr=qC�j=                                    Bxk��  �          @c33@(�ÿ�ff?\A��C���@(���
=q?p��Au�C�U�                                    Bxk�&�  �          @hQ�@*=q����?�{A���C���@*=q�\)?�  A���C���                                    Bxk�54  �          @aG�@(�ÿ�ff?�
=A��C��3@(�����?Y��A`(�C�y�                                    Bxk�C�  �          @]p�@'
=����?���AݮC���@'
=���H?�=qA��
C���                                    Bxk�R�  �          @]p�@$z��(�?��AӅC�!H@$z��?xQ�A��RC�q�                                    Bxk�a&  �          @\��@*�H��=q?�p�Ȁ\C���@*�H����?uA���C��3                                    Bxk�o�  �          @\(�@,(���p�?�G�A�33C���@,(�����?�G�A���C���                                    Bxk�~r  �          @a�@(Q���?�  A��C�ٚ@(Q��p�?�p�A�G�C��                                     BxkǍ  �          @aG�@.{��=q?��
A�=qC��@.{��ff?���A�  C�C�                                    BxkǛ�  �          @^{@2�\���?���AǙ�C���@2�\��G�?z�HA�{C��                                     BxkǪd  �          @]p�@333�У�?�
=A�{C���@333��33?&ffA-C���                                    Bxkǹ
  �          @^�R@=p����?�p�A��\C��@=p���\)?J=qAS
=C��                                    Bxk�ǰ  �          @]p�@?\)��G�?�
=A�\)C���@?\)��ff?@  AI��C�Q�                                    Bxk��V  �          @\��@HQ�W
=?�A�ffC�q�@HQ쿓33?\(�Ah��C��                                    Bxk���  �          @aG�@J=q���
?�\)A�33C���@J=q����?E�AI�C���                                    Bxk��  �          @aG�@:�H��  ?�  A�\)C�k�@:�H��ff?@  AF=qC�'�                                    Bxk�H  �          @b�\@333��ff?�z�A�(�C��f@333��
?z�A�RC�Ф                                    Bxk��  �          @dz�@7
=��?�{A�\)C���@7
=��
?�A=qC�{                                    Bxk��  �          @c33@E��G�?�Q�A�\)C��f@E�Ǯ?B�\AF�HC��)                                    Bxk�.:  T          @_\)@Mp��Y��?��\A��C���@Mp���\)?8Q�A@Q�C�k�                                    Bxk�<�  �          @dz�@QG��L��?�z�A��\C�%@QG���{?^�RAb�\C��H                                    Bxk�K�  �          @e@Fff��z�?���A���C���@Fff�\?xQ�Az�HC���                                    Bxk�Z,  T          @dz�@@�׿�ff?�33A�G�C�Q�@@�׿�z�?s33AvffC��                                    Bxk�h�  �          @dz�@C33����?��A�33C�W
@C33���?Tz�AW�C��q                                    Bxk�wx  �          @e�@E���=q?��RA�ffC�Q�@E����?G�AJ=qC��
                                    BxkȆ  �          @c�
@HQ쿗
=?���A�G�C��3@HQ쿽p�?G�AK�
C�T{                                    BxkȔ�  �          @j�H@S33����?�
=A���C�H@S33����?L��AJ�\C���                                    Bxkȣj  �          @k�@QG���?�(�A�G�C�  @QG����R?O\)AK\)C��H                                    BxkȲ  �          @mp�@W
=�}p�?�p�A�Q�C��3@W
=���?^�RAXz�C�W
                                    Bxk���  �          @k�@Tz῀  ?��RA��
C���@TzῪ=q?aG�A\��C��                                    Bxk��\  �          @l��@W��xQ�?�
=A��RC���@W����
?Tz�AN�HC��{                                    Bxk��  �          @n�R@Z=q�fff?�(�A��HC���@Z=q��p�?c�
A\  C��                                    Bxk��  �          @n{@Y���Q�?��RA�ffC�B�@Y����z�?n{Ah��C��q                                    Bxk��N  �          @j�H@]p����?���A���C�'�@]p��J=q?n{Ak�
C��\                                    Bxk�	�  T          @mp�@\�ͿG�?�{A��C��
@\�Ϳ��?O\)AJ�HC�AH                                    Bxk��  �          @j=q@W
=�fff?�33A��HC�}q@W
=���H?Q�AO
=C�3                                    Bxk�'@  �          @k�@Z=q�L��?�\)A��
C�g�@Z=q����?Q�AM�C�                                    Bxk�5�  �          @j�H@W
=�xQ�?���A�G�C��3@W
=��G�?=p�A;33C��
                                    Bxk�D�  �          @k�@U�xQ�?�(�A�ffC��=@U��ff?\(�AW�C�aH                                    Bxk�S2  �          @i��@Q녿k�?�ffA�C�/\@Q녿��
?uAs33C�]q                                    Bxk�a�  �          @k�@N�R��z�?���A�C��@N�R��G�?c�
A`Q�C�w
                                    Bxk�p~  �          @hQ�@E����?��\A��\C��@E���H?G�AFffC���                                    Bxk�$  �          @j=q@C�
����?��RA�ffC�o\@C�
���?0��A-p�C�>�                                    Bxkɍ�  �          @j�H@@�׿�
=?�(�A�\)C�t{@@�׿�(�?!G�A33C�e                                    Bxkɜp  �          @k�@?\)��  ?�p�A�=qC���@?\)��p�?�  A���C���                                    Bxkɫ  �          @l(�@C�
��=q?���A�  C�8R@C�
��G�?�=qA�C�
=                                    Bxkɹ�  �          @k�@C33����?��A�p�C�˅@C33��ff?��
A�\)C���                                    Bxk��b  �          @j�H@>�R���?�p�A�{C�U�@>�R��
=?k�Ag�
C��\                                    Bxk��  �          @hQ�@<(���{?��A�ffC��H@<(���(�?O\)AO33C�R                                    Bxk��  �          @i��@@�׿�?��HA�
=C�~�@@�׿�(�?�RA��C�l�                                    Bxk��T  �          @j=q@G���\)?�  A
=C�K�@G�����>�G�@�p�C��f                                    Bxk��  
�          @h��@O\)���R?(��A(z�C���@O\)��\)>#�
@�RC��3                                    Bxk��  �          @i��@Q녿���?^�RA\(�C�@Q녿��
>���@�33C���                                    Bxk� F  �          @h��@XQ쿋�?Q�AO�C�{@XQ쿥�>�(�@أ�C���                                    Bxk�.�  �          @hQ�@W���ff?Q�AR{C�b�@W���  >�ff@��HC���                                    Bxk�=�  �          @i��@U��Q�?W
=AT(�C�.@U��33>��@ϮC���                                    Bxk�L8  �          @j=q@Vff���?z�HAw�C���@Vff����?z�A��C�                                    Bxk�Z�  �          @j=q@Y���h��?}p�A|z�C�|)@Y����
=?&ffA#�
C�g�                                    Bxk�i�  �          @l(�@`  �5?fffAa�C�Ff@`  �u?�RA�
C�T{                                    Bxk�x*  T          @k�@^�R�}p�?�RAz�C��@^�R���>�=q@��C��\                                    Bxkʆ�  �          @mp�@X�ÿ�
=>\@��HC��3@X�ÿ�p����Ϳ�\)C�AH                                    Bxkʕv  �          @j�H@Y����\)=�G�?�Q�C��@Y����=q��{��=qC�K�                                    Bxkʤ  �          @j=q@P�׿�논��
�uC�� @P�׿�ff����	��C�J=                                    Bxkʲ�  �          @j�H@H�ÿ�=�G�?�Q�C��=@H�ÿ��
�   ��=qC�:�                                    Bxk��h  �          @l��@7���?��A�=qC�Z�@7��G�>�p�@�33C��=                                    Bxk��  �          @l��@%��ff?�\)A���C��)@%��%>���@��C�z�                                    Bxk�޴  �          @l(�@,���z�?s33Ao�
C���@,���   >��@��C��{                                    Bxk��Z  �          @n�R@#33�ff?���A�Q�C���@#33�*=q?�\@��
C�޸                                    Bxk��   �          @p  @!����?�{A�=qC�W
@!��,��?   @�{C��{                                    Bxk�
�  �          @k�@,����\?p��AnffC�޸@,���{>\)@��C�Ǯ                                    Bxk�L  �          @l(�@+���?z�HAw33C��H@+��!�>#�
@!�C�^�                                    Bxk�'�  �          @p  @6ff���?�G�AzffC�(�@6ff�=q>k�@`��C���                                    Bxk�6�  �          @n�R@7��{?Tz�AN�HC�%@7���=u?Y��C�:�                                    Bxk�E>  �          @p  @3�
��\?uAn{C�aH@3�
��R>��@�C�B�                                    Bxk�S�  �          @o\)@%�� ��?}p�AuC��f@%��,��=���?˅C��H                                    Bxk�b�  �          @n�R@-p��
=?}p�Aw33C��@-p��#33>#�
@��C�aH                                    Bxk�q0  �          @j�H@&ff�=q?n{AjffC��)@&ff�%�=��
?�C��H                                    Bxk��  �          @j�H@$z���?z�HAw�C�XR@$z��'
==�?���C�Ff                                    Bxkˎ|  �          @hQ�@,(����?��\A�=qC�]q@,(��=q>aG�@b�\C�\                                    Bxk˝"  �          @i��@>�R���?�=qA�=qC���@>�R�33>�
=@��C�                                    Bxk˫�  �          @mp�@E���=q?}p�Av�\C���@E��33>��
@��C�"�                                    Bxk˺n  �          @o\)@>�R� ��?��
A�
C��{@>�R�\)>���@�G�C��                                     Bxk��  �          @p  @@  ���R?��
A�Q�C�>�@@  �{>��R@��RC�                                    Bxk�׺  �          @q�@:=q��?���A���C���@:=q�Q�>�33@��C�`                                     Bxk��`  �          @n{@+���\?�
=A�C��f@+��#33>�{@��RC�9�                                    Bxk��  �          @n{@���p�?��A�
=C�k�@���1G�>�(�@��C��)                                    Bxk��  �          @mp�@p��
�H?ٙ�A��HC�K�@p��'�?\(�AXQ�C���                                    Bxk�R  �          @k�@{�G�?���A���C���@{�(Q�?
=AC���                                    Bxk� �  �          @h��@
=�Q�?�z�A���C�j=@
=�-p�?�A�
C���                                    Bxk�/�  �          @k�@{��?�{A�C��@{�,(�>�@�p�C�@                                     Bxk�>D  �          @Z=q@���  ?��\A��C�4{@���#33>�(�@�Q�C�n                                    Bxk�L�  �          @l(�@!���?���A�(�C�@!��*=q>8Q�@2�\C��\                                    Bxk�[�  �          @g�@
=��?��A���C�3@
=�.{>\@���C�q�                                    Bxk�j6  �          @h��@p��#33?�\)A��RC�p�@p��7
=>��@�Q�C��\                                    Bxk�x�  �          @i��@G��%?���A�z�C��R@G��>{?
=AQ�C��                                    Bxk̇�  �          @mp�?�{�-p�?�Q�A�=qC�9�?�{�G�?(��A$z�C�]q                                    Bxk̖(  �          @mp�?��H�1G�?�G�A���C�ٚ?��H�Mp�?333A-�C��                                    Bxk̤�  �          @j=q?�{�0��?��
A��HC�*=?�{�L��?8Q�A5G�C�^�                                    Bxk̳t  �          @u?�=q�5?���A��C�l�?�=q�R�\?8Q�A-p�C��                                    Bxk��  �          @u@�
�,(�?�AᙚC��)@�
�I��?E�A9p�C��
                                    Bxk���  �          @s33@  �%?�33A�p�C���@  �@  ?&ffA��C�q�                                    Bxk��f  �          @k�@\)�&ff?�ffA�{C�S3@\)�8��>��R@�p�C��R                                    Bxk��  �          @i��@
=�.�R?�z�A��HC���@
=�>{>\)@
�HC���                                    Bxk���  T          @dz�@(��#�
?��A��C�AH@(��333>.{@0��C�                                    Bxk�X  �          @b�\@z��\)?uAz�HC���@z��*=q<�>�
=C���                                    Bxk��  �          @aG�@=q���?^�RAfffC���@=q�#33���
��p�C��                                    Bxk�(�  �          @dz�@"�\�?aG�Aep�C���@"�\�   <#�
=�Q�C��R                                    Bxk�7J  �          @c33@%��z�?E�AHz�C��@%��(���Q��G�C�Ff                                    Bxk�E�  �          @dz�@-p����?&ffA)�C�xR@-p���\�.{�*�HC��=                                    Bxk�T�  �          @`  @.�R�	��>\@ȣ�C��)@.�R�	�������ӅC��                                    Bxk�c<  �          @_\)@0����>�z�@�Q�C�7
@0�������(�C�p�                                    Bxk�q�  �          @tz�@���-p�?�\)A�Q�C��@���;�=�Q�?��C��{                                    Bxk̀�  �          @s33@:=q�
=?��A=qC�|)@:=q������
��33C�<)                                    Bxk͏.  �          @s�
@B�\��?5A,(�C�.@B�\��\��G���z�C��H                                    Bxk͝�  �          @w
=@L���>�@�RC�h�@L���Q쾔z����C�0�                                    Bxkͬz  �          @s33@I���>�Q�@���C�7
@I���������
=C�G�                                    Bxkͻ   �          @vff@H���
�H>u@g
=C��R@H���
=����  C�3                                    Bxk���  �          @vff@G����?��A�C�˅@G��(���  �tz�C�u�                                    Bxk��l  �          @u@Tz��z�>�\)@��C��@Tz��׾�(���(�C�@                                     Bxk��  �          @w
=@K����>\@�ffC��@K��Q������
C��                                    Bxk���  �          @vff@Fff�G�<�>�G�C���@Fff���J=q�=p�C�˅                                    Bxk�^  �          @u@<����;#�
�(�C�)@<����R����z�RC�q�                                    Bxk�  �          @tz�@B�\�G�>�
=@��HC���@B�\�G���(���p�C��                                     Bxk�!�  �          @s�
@I�����R?L��A@��C�޸@I�����<��
>�{C��                                    Bxk�0P  �          @s33@@  �	��?^�RATz�C�%@@  �z�<�>�C��                                    Bxk�>�  �          @q�@J�H��33?O\)AD��C��\@J�H��
=u?s33C��                                     Bxk�M�  �          @r�\@S�
����?h��A^{C��@S�
����>��@�Q�C��R                                    Bxk�\B  �          @r�\@QG���  ?B�\A8z�C���@QG���33=�\)?��
C��                                    Bxk�j�  �          @q�@L(���Q�?z�A��C�S3@L(��G��.{�%C��3                                    Bxk�y�  
�          @q�@Mp���z�?�\@���C��)@Mp���(��aG��Z=qC�AH                                    BxkΈ4  �          @q�@AG����?333A+�
C�T{@AG��  ���   C���                                    BxkΖ�  �          @tz�@(Q��"�\?���A�{C��@(Q��0��=�Q�?�33C���                                    BxkΥ�  �          @q�@6ff�{?���A���C�3@6ff�p�>L��@EC��
                                    Bxkδ&  �          @p  @ ���.�R?\)A	p�C�E@ ���/\)���H��{C�1�                                    Bxk���  �          @n{@�R�*=q?0��A,��C��f@�R�.{��{��ffC�(�                                    Bxk��r  �          @s33@Q��:=q?��A��C��f@Q��9���\)�Q�C���                                    Bxk��  �          @n�R@
=�B�\>���@ÅC�c�@
=�>{�B�\�<  C��{                                    Bxk��  �          @hQ�@ff�'�?!G�A!C��
@ff�*=q�Ǯ�ǮC��R                                    Bxk��d  �          @j�H@8���ff?Q�AO33C�H@8����׼��
��=qC�f                                    Bxk�
  �          @j�H@4z����?@  A<z�C�f@4z��z����p�C�H�                                    Bxk��  �          @q�@5�=q?+�A#�
C��
@5�\)��=q����C�e                                    Bxk�)V  �          @s33@5���R?\)A�HC�` @5�� �׾����C�5�                                    Bxk�7�  �          @u@333�$z�?�\@��RC��R@333�%�����Q�C��\                                    Bxk�F�  �          @x��@4z��)��>�z�@�33C�b�@4z��$z�8Q��+33C���                                    Bxk�UH  �          @w�@2�\�(Q�>��H@��C�L�@2�\�(Q�
=q��
=C�XR                                    Bxk�c�  �          @tz�@.�R�(��>Ǯ@�(�C��@.�R�%�!G���
C�B�                                    Bxk�r�  �          @o\)@,(��"�\>�@��
C�L�@,(��!녿���RC�`                                     Bxkρ:  	`          @p��@/\)�#33>�  @p  C���@/\)��Ϳ:�H�4��C�R                                    BxkϏ�  �          @p  @A���?�\@�=qC�}q@A��
=q������33C�E                                    BxkϞ�  �          @l(�@=p��G�?L��AJ=qC��@=p������
��33C�Ф                                    Bxkϭ,  �          @mp�?���L��<�>�
=C�>�?���>{��Q���\)C�1�                                    Bxkϻ�  �          @l��@��8Q�>�Q�@�33C�.@��333�B�\�>�\C���                                    Bxk��x  �          @n{@'��$z�?\)A�C��q@'��%����C��H                                    Bxk��  �          @j�H@(Q��!G�>�@�
=C�'�@(Q�� �׿����C�33                                    Bxk���  �          @qG�@�R�AG������C�9�@�R�/\)������HC���                                    Bxk��j  �          @r�\@{�Dz�=#�
?
=C��@{�6ff�����  C��3                                    Bxk�  �          @u@%�4z�=�?��C�L�@%�)���xQ��j�\C�/\                                   Bxk��  s          @~{@.�R�7�>��
@��
C�˅@.�R�1G��O\)�;33C�J=                                    Bxk�"\  �          @~{@3�
�1G�>��R@�{C���@3�
�+��G��6=qC�9�                                    Bxk�1  �          @z=q@)���7�=��
?�{C�aH@)���+�����uC�`                                     Bxk�?�  	�          @|��@'
=�<�;L���;�C���@'
=�(�ÿ������HC�T{                                    Bxk�NN  �          @|��@.{�6ff��Q쿡G�C��\@.{�&ff��
=���C�&f                                    Bxk�\�  �          @{�@1G��2�\=L��?8Q�C�n@1G��%���
�s�
C�|)                                    Bxk�k�  �          @|��@.�R�7
=<�>��C���@.�R�(�ÿ�=q�~=qC���                                    Bxk�z@  �          @���@2�\�9���.{��C��)@2�\�'
=���
��{C��H                                    BxkЈ�  �          @��\@7��8�þ�\)�~�RC�aH@7��#33�������C�'�                                    BxkЗ�  �          @��@:=q�3�
����o\)C���@:=q�\)�����ffC���                                    BxkЦ2  �          @�33@B�\�0  <��
>��RC���@B�\�"�\��ff�n{C��                                    Bxkд�  �          @�Q�@<(��.{���
���C���@<(��   ��=q�z=qC��
                                    Bxk��~  �          @��\@:�H�3�
��p����C�@:�H�(���Q���=qC�f                                    Bxk��$  �          @��\@6ff�8Q���ٙ�C�T{@6ff�p���=q���C���                                    Bxk���  �          @��@333�8Q�z���RC��@333��H��z����C��\                                    Bxk��p  �          @�Q�@,(��:�H�����{C�P�@,(��p���z��Ù�C���                                    Bxk��  �          @w
=@.�R�)��>���@�\)C���@.�R�#33�E��:�RC�q�                                    Bxk��  �          @qG�@(Q��#33?\(�ATQ�C��@(Q��+��W
=�QG�C�1�                                    Bxk�b  T          @p��@!��)��?Tz�AK�C��q@!��0�׾�z�����C�Ff                                    Bxk�*  �          @r�\@%�,��?
=A�C��@%�-p������C��)                                    Bxk�8�  �          @u@%��0  ?#�
AffC���@%��1녿   ���C�p�                                    Bxk�GT  T          @u�@#33�1�?+�A ��C�<)@#33�4z���H����C�                                    Bxk�U�  �          @qG�@   �1�>�@�ffC�  @   �.�R�0���(��C�@                                     Bxk�d�  T          @qG�@ff�1�?p��Ag33C��@ff�;���  �r�\C�h�                                    Bxk�sF  �          @p��@\)�.�R?.{A%�C�%@\)�1녾����
=C���                                    Bxkс�  �          @o\)@,���!�>�G�@��HC�c�@,���   �(��{C���                                    Bxkѐ�  �          @vff@<(��p�>aG�@VffC�
@<(���G��;33C���                                    Bxkџ8  �          @w
=@B�\��>aG�@N{C�
=@B�\�  �@  �3�
C���                                    Bxkѭ�  	`          @w
=@B�\��>�  @l(�C�@B�\�G��:�H�-p�C��q                                    BxkѼ�  �          @xQ�@A����>aG�@QG�C��=@A���\�E��6ffC�w
                                    Bxk��*  �          @z�H@B�\�p�>�  @e�C���@B�\�ff�E��4��C�'�                                    Bxk���  �          @z�H@@���   >#�
@�C�q@@���
=�\(��K�C��R                                    Bxk��v  �          @�  @G
=�\)>��
@��C���@G
=����5�#�C�'�                                    Bxk��  �          @~{@HQ��=q>�(�@��C�7
@HQ����
=��
C�h�                                    Bxk��  �          @y��@G��33>�G�@�  C��\@G��녿����C��                                    Bxk�h  �          @u@HQ��	��?   @���C�@HQ��
�H��
=��
=C��                                    Bxk�#  �          @vff@J=q�Q�?   @�
=C���@J=q�	���������C��H                                    Bxk�1�  �          @z=q@H���\)?��Ap�C�>�@H����\��p���(�C��3                                    Bxk�@Z  �          @p��@Mp���\)?z�A
=C��@Mp���Q�k��`  C�g�                                    Bxk�O   �          @tz�@P  ���?&ffAG�C�޸@P  ���R�8Q��-p�C�7
                                    Bxk�]�  �          @tz�@R�\���
?G�A<  C��@R�\��Q켣�
��z�C��)                                    Bxk�lL  �          @s�
@L(���z�?J=qA?\)C��f@L(��z὏\)��G�C��f                                    Bxk�z�  �          @u@;����?�RAQ�C�` @;�����
=�ə�C�!H                                    Bxk҉�  �          @|(�@:�H�#�
?0��A"ffC�c�@:�H�'
=��
=�ÅC��                                    BxkҘ>  �          @z�H@<���{?=p�A/
=C��@<���#�
��{���C��=                                    BxkҦ�  �          @z=q@>{���?Tz�AEG�C���@>{�!녾k��XQ�C��=                                    Bxkҵ�  �          @{�@Fff�z�?#�
A�C���@Fff�Q�\���HC�=q                                    Bxk��0  T          @{�@H���G�?�RA{C�3@H���z᾽p����RC��H                                    Bxk���  �          @|(�@HQ��  ?W
=AEC�)@HQ�����.{��HC�=q                                    Bxk��|  �          @x��@>�R�33?s33Aa�C�%@>�R�\)���Ϳ�z�C��                                    Bxk��"  �          @y��@B�\���?n{A\Q�C��@B�\�(����Ϳ�C��q                                    Bxk���  �          @|(�@Fff�  ?s33A^{C�@Fff�(����
��\)C��                                    Bxk�n  �          @~{@J=q��?��HA��C�W
@J=q���>k�@S�
C�q�                                    Bxk�  �          @}p�@G���?�
=A���C��@G���H>B�\@+�C��                                    Bxk�*�  �          @{�@G��
�H?xQ�Ad(�C���@G��Q����C�^�                                    Bxk�9`  �          @{�@H���{?Tz�AB=qC�e@H���
=�8Q��'
=C���                                    Bxk�H  �          @z�H@H���{?B�\A3
=C�e@H�����u�aG�C���                                    Bxk�V�  �          @|(�@L���
�H?L��A:�\C���@L���33�B�\�/\)C��                                    Bxk�eR  �          @|(�@HQ���
?#�
A{C��H@HQ����Ǯ��ffC�n                                    Bxk�s�  �          @}p�@G��
=?&ffA�HC�t{@G��=q�����(�C�#�                                    Bxkӂ�  �          @{�@C33�Q�?+�AG�C�@C33�(��������C���                                    BxkӑD  �          @|(�@C�
���?&ffAQ�C��
@C�
��;�
=��p�C���                                    Bxkӟ�  �          @}p�@Fff��?.{AffC�C�@Fff�(��Ǯ��z�C���                                    BxkӮ�  �          @|(�@Dz���?&ffA�C�&f@Dz���H��
=��z�C���                                    Bxkӽ6  �          @z�H@Dz��ff?(��A�C�Q�@Dz��=q�������C��R                                    Bxk���  �          @z�H@G���?&ffA��C��@G���\���\C���                                    Bxk�ڂ  �          @}p�@N{���?333A#\)C���@N{��\������(�C�H�                                    Bxk��(  �          @|(�@O\)�
=q?z�A�HC�33@O\)��;�������C��{                                    Bxk���  �          @|(�@Q��Q�>�ff@�  C���@Q�����\����C��R                                    Bxk�t  T          @|(�@S33�ff>��H@�{C��=@S33�
=��ff��=qC���                                    Bxk�  �          @{�@N{�
=q?�RA��C�
=@N{�{��p�����C��{                                    Bxk�#�  �          @~�R@O\)�(�?:�HA)G�C��)@O\)��\��\)����C�aH                                    Bxk�2f  �          @}p�@R�\���?
=q@�p�C�}q@R�\�
=q��(���C�S3                                    Bxk�A  �          @|(�@S�
��
?�A  C��@S�
�ff�\��\)C��                                    Bxk�O�  �          @~{@Q��
=q?�A\)C�T{@Q��(���
=��=qC�                                      Bxk�^X  �          @|(�@QG��
=q>��@�z�C�E@QG��	����\��C�P�                                    Bxk�l�  T          @y��@P  �
=>��@�G�C���@P  ���
=q�   C��{                                    Bxk�{�  �          @w�@N{�ff?   @�p�C�s3@N{�
=������C�b�                                    BxkԊJ  �          @w�@J=q�(�>�(�@�z�C��@J=q�
=q�\)�p�C��3                                    BxkԘ�  �          @|��@N�R���?:�HA+�C�@ @N�R�  ��=q�x��C���                                    Bxkԧ�  �          @vff@J=q�
=?
=AC�*=@J=q�	���\��ffC��q                                    BxkԶ<  �          @vff@HQ��Q�?(��Ap�C��f@HQ���;�{��  C�q�                                    Bxk���  �          @u�@Dz���?&ffA��C�T{@Dz��\)��p���\)C��                                    Bxk�ӈ  �          @z=q@@���
=?O\)A?33C���@@���{������p�C�H�                                    Bxk��.  �          @u@E�	��?@  A4��C��@E��׾���|��C��q                                    Bxk���  T          @l(�@%�   ?+�A(��C���@%�"�\���H���C��                                     Bxk��z  
�          @dz�@�
�6ff?�AC��@�
�2�\�G��L  C�<)                                    Bxk�   �          @e?�\)�AG�=���?�33C���?�\)�0�׿�p�����C��                                    Bxk��  �          @^�R?�  �>�R>W
=@`  C�1�?�  �1G���\)��
=C�!H                                    Bxk�+l  �          @^�R?���7�?   A�C��R?���333�Q��ZffC���                                    Bxk�:  �          @Q�?����<��=u?��C���?����+����R��p�C��q                                    Bxk�H�  �          @QG�?����7
=>.{@=p�C�Y�?����(�ÿ�{���C�T{                                    Bxk�W^  �          @P��?��H�:=q>\)@!�C�O\?��H�*�H��z���ffC�N                                    Bxk�f  T          @QG�?�p��8��>�=q@���C��{?�p��-p����
���C�T{                                    Bxk�t�  �          @^{?��
�<(�>�Q�@���C��
?��
�333�z�H��G�C�:�                                    BxkՃP  �          @j=q@Q��<(�>��@�C��\@Q��6ff�c�
�_�
C�e                                    BxkՑ�  �          @j�H@���5?�\@�
=C�=q@���1G��O\)�K�C��
                                    Bxkՠ�  �          @i��@33�2�\?   @���C���@33�.{�L���JffC��                                    BxkկB  T          @u�@���7�?:�HA0��C��)@���8�ÿ�R��\C��)                                    Bxkս�  �          @}p�@,���4z�?5A$  C��@,���5���R��C��=                                    Bxk�̎  �          @xQ�@/\)�,(�?z�A	�C��=@/\)�*=q�.{�!G�C��=                                    Bxk��4  �          @y��@2�\�(Q�?(��A��C�]q@2�\�)���z��
{C�C�                                    Bxk���  �          @vff@333�!G�?=p�A2{C�f@333�%������ffC���                                    Bxk���  �          @x��@9���!G�?��A{C���@9���!G��
=�(�C��                                    Bxk�&  �          @tz�@(���,(�?�@��HC�<)@(���(Q�=p��4Q�C��                                    Bxk��  �          @j=q@Q��,(�?!G�A�\C��)@Q��,(��&ff�$(�C�                                    Bxk�$r  �          @i��@   �&ff>��@���C��\@   �!녿@  �>�HC�O\                                    Bxk�3  T          @]p�@��\)>Ǯ@У�C���@�����E��N=qC�O\                                    Bxk�A�  �          @[�@��"�\>�33@��HC��
@���H�Tz��a�C��H                                    Bxk�Pd  T          @\(�@�H�(�>�@p�C�c�@�H��R�z�H����C���                                    Bxk�_
  �          @_\)@�H�   >#�
@+�C�@�H�33�z�H��\)C�9�                                    Bxk�m�  �          @j�H@!��'
=>�@���C�@!��"�\�@  �=C�k�                                    Bxk�|V  �          @j�H@p��*�H?\)A�C�Y�@p��(Q�5�3\)C���                                    Bxk֊�  �          @k�@�R�*=q?��A��C�~�@�R�(�ÿ+��(��C��R                                    Bxk֙�  �          @n�R@\)�-p�?z�A�RC�H�@\)�+��8Q��1C�u�                                    Bxk֨H  �          @mp�@��-p�?(�A\)C��\@��,(��0���-p�C��                                    Bxkֶ�  �          @j=q@#�
���?��
A��C�q�@#�
�&ff�.{�.�RC�B�                                    Bxk�Ŕ  �          @n{@{��R?�(�A�33C�~�@{�0�׽L�Ϳ=p�C��3                                    Bxk��:  �          @r�\@���333?h��A_33C�:�@���:=q������C���                                    Bxk���  �          @qG�@%�&ff?^�RAV=qC�e@%�.{������C�˅                                    Bxk��  �          @hQ�@���\)?s33AtQ�C�@ @���)����z���z�C�\)                                    Bxk� ,  �          @j=q@)����\?�G�A�ffC���@)���   ����=qC�^�                                    Bxk��  T          @n{@,(���?�z�A�C��)@,(��#�
���
�uC�8R                                    Bxk�x  �          @j�H@p��{?�\)A���C�w
@p��-p��\)�	��C�#�                                    Bxk�,  �          @c33@{�'
=?k�Ao�C�#�@{�/\)�Ǯ��G�C�w
                                    Bxk�:�  �          @g
=@33� ��?���A�G�C�H�@33�0  ����Q�C��)                                    Bxk�Ij  �          @j=q@����?�p�A���C���@���.{�����C���                                    Bxk�X  �          @g�@����?�p�A�=qC���@��,(����
���C�                                    Bxk�f�  �          @h��@\)�Q�?�
=A��C�"�@\)�*=q�u�aG�C���                                    Bxk�u\  �          @e@��G�?��A��C���@��(Q�>\)@��C�b�                                    Bxkׄ  �          @g�@*=q���?uAuC�˅@*=q��;L���I��C��=                                    Bxkג�  �          @h��@.�R�ff?�
=A���C�5�@.�R�=q=�\)?���C�K�                                    BxkסN  �          @i��@.�R�{?z�HAyp�C�q�@.�R��H�#�
�!�C�7
                                    Bxkׯ�  �          @fff@)����R?xQ�Az�RC���@)�����8Q��4z�C���                                    Bxk׾�  �          @e@&ff���?�  A��C�}q@&ff�{�.{�*=qC�>�                                    Bxk��@  T          @dz�@   �Q�?^�RAc\)C�>�@   � �׾�����z�C�u�                                    Bxk���  �          @c33@���+�?B�\AF=qC���@���.{�z����C�s3                                    Bxk��  �          @^�R?�(��1G�?@  AG33C��{?�(��333�!G��'
=C��\                                    Bxk��2  �          @_\)?�Q��5�?+�A/�
C�0�?�Q��3�
�:�H�B=qC�E                                    Bxk��  �          @b�\@ ���5�?+�A.ffC��=@ ���4z�=p��@  C���                                    Bxk�~  �          @aG�@33�333?\)A�C�)@33�.�R�O\)�VffC�k�                                    Bxk�%$  �          @b�\@���/\)?�RA#
=C��@���-p��:�H�?�C�*=                                    Bxk�3�  �          @e@���'
=?�\AffC�J=@���"�\�G��H��C��=                                    Bxk�Bp  �          @g�@G��0  ?
=qA	��C�Ǯ@G��+��Q��Qp�C�#�                                    Bxk�Q  �          @g�@��,(�?�RA��C�s3@��*�H�8Q��8��C��R                                    Bxk�_�  �          @j�H@��(��?c�
Aa��C��@��0  ����
=C�Z�                                    Bxk�nb  �          @r�\@p��3�
?5A,��C���@p��3�
�333�)�C��
                                    Bxk�}  �          @s�
@\)�2�\?0��A(  C��f@\)�2�\�5�+33C��                                    Bxk؋�  �          @s33@-p���R?n{Ac33C��)@-p��(Q쾸Q���(�C���                                    BxkؚT  �          @qG�@6ff�
�H?�z�A�\)C�` @6ff�p��#�
��Q�C��q                                    Bxkب�  �          @qG�@5���?�z�A�C�33@5����>�\)@�\)C��\                                    Bxkط�  �          @p  @5���?�(�A�z�C���@5��(�=�\)?��C���                                    Bxk��F  �          @p��@1��	��?�=qA�\)C�,�@1��!G�>��@\)C���                                    Bxk���  �          @p��@+���?z�HAqG�C��@+��'
=�������C��                                    Bxk��  �          @o\)@*�H���?���A���C��@*�H�'
=�B�\�=p�C��f                                    Bxk��8  �          @l��@1��z�?��\A�\)C��\@1���>�@C�u�                                    Bxk� �  �          @j=q@'��  ?��A��RC���@'��!G���\)���C�                                    Bxk��  �          @i��@ff�*=q?Q�AO\)C��3@ff�.�R�����
C�Y�                                    Bxk�*  �          @h��@���!�?h��Ag33C�f@���)�������Q�C�T{                                    Bxk�,�  �          @e@Q��!�?aG�Ad��C��R@Q��)����(��޸RC��
                                    Bxk�;v  �          @b�\@33�!G�?n{At��C�0�@33�*=q�Ǯ��G�C�l�                                    Bxk�J  T          @e�@ ���?�G�A���C�z�@ ���"�\�u�uC�T{                                    Bxk�X�  �          @c�
@2�\����?�ffA��C�z�@2�\�{�#�
�aG�C���                                    Bxk�gh  �          @aG�@-p���?W
=A\��C�J=@-p���R�����  C�T{                                    Bxk�v  �          @^�R@'��Q�?Tz�A]p�C�q�@'��G���z����C���                                    Bxkل�  �          @`  @'
=���?E�AJ�\C���@'
=�33�Ǯ��=qC�XR                                    BxkٓZ  �          @]p�@&ff���R?��A��C�L�@&ff�G�    �#�
C�s3                                    Bxk٢   �          @_\)@#�
� ��?�p�A��C���@#�
�ff=�G�?��C���                                    Bxkٰ�  �          @]p�@�R�\)?h��At(�C��@�R�=q��\)��
=C��=                                    BxkٿL  �          @]p�@!G��
�H?p��A}C��H@!G��
=�aG��j�HC�t{                                    Bxk���  �          @\(�@#�
�ff?h��Av�RC�S3@#�
�녾aG��g
=C�(�                                    Bxk�ܘ  T          @[�@"�\�
=?k�Ay��C�'�@"�\��\�W
=�e�C���                                    Bxk��>  �          @^�R@'��
=?^�RAf�HC��@'��G���=q����C��                                    Bxk���  �          @aG�@%���R?\(�Aa�C���@%�����{��
=C��q                                    Bxk��  �          @a�@(Q����?Y��A_\)C�\@(Q������
��\)C�4{                                    Bxk�0  T          @c33@,����?\(�Aa��C���@,���녾�\)��  C��                                    Bxk�%�  �          @c�
@8�ÿ�(�?#�
A&ffC��@8���녾�������C�t{                                    Bxk�4|  �          @c33@<�Ϳ�\)?!G�A"ffC��R@<�Ϳ�Q쾸Q����
C�`                                     Bxk�C"  �          @fff@<�Ϳ�z�?B�\ABffC��@<����\��=q���\C���                                    Bxk�Q�  �          @fff@(����?\(�A\z�C��
@(���=q��p����HC�Ф                                    Bxk�`n  �          @e@!G����?h��Aj�RC�>�@!G��!녾�p���\)C�o\                                    Bxk�o  �          @fff@�H�   ?s33At��C��@�H�(�þǮ�ǮC�9�                                    Bxk�}�  �          @g�@�H�!G�?p��AqC��@�H�*=q�����  C�(�                                    Bxkڌ`  �          @hQ�@��� ��?��A�33C�Ф@���,�;�������C�˅                                    Bxkڛ  �          @i��@p��{?���A�z�C�h�@p��+���\)��z�C�B�                                    Bxkک�  �          @h��@{��R?�G�A���C�l�@{�*=q��{���C�p�                                    BxkڸR  T          @^�R@%�
=?fffAq�C�o\@%�녾�  ���C�W
                                    Bxk���  T          @Tz�@5��\?Y��Am��C��@5��޸R    ��\)C�7
                                    Bxk�՞  �          @^{@AG���ff?!G�A(��C�c�@AG���z�W
=�a�C��3                                    Bxk��D  �          @a�@=p���?&ffA)G�C�E@=p���33������=qC��\                                    Bxk���  �          @e�@C�
��  ?#�
A%��C�&f@C�
���������C���                                    Bxk��  �          @dz�@A녿��
?��A��C�˅@A녿���Q�����C�]q                                    Bxk�6  �          @b�\@AG���  ?\)A��C��@AG���ff�\�ƸRC���                                    Bxk��  �          @e@J=q���>�
=@�ffC�P�@J=q�У׾�G���(�C�]q                                    Bxk�-�  �          @c�
@HQ��ff?.{A0  C�ٚ@HQ��
=�8Q��4z�C��                                    Bxk�<(  �          @i��@J=q��z�?:�HA9��C�!H@J=q���B�\�:�HC��                                    Bxk�J�  �          @j�H@J�H��p�?(��A$��C���@J�H���þ�\)��ffC��                                    Bxk�Yt  �          @c�
@E���z�?#�
A&=qC���@E���G���=q��G�C�                                      Bxk�h  �          @e@C�
�ٙ�?B�\AD��C�u�@C�
��{�8Q��:=qC�aH                                    Bxk�v�  �          @c�
@K���Q�?.{A/�C���@K���=q��� ��C��f                                    Bxkۅf  �          @b�\@N{���\?0��A5G�C�8R@N{���������HC���                                    Bxk۔  �          @X��@E����R?333A=��C�{@E����#�
�#�
C��                                    Bxkۢ�  �          @n�R@S33���?E�A?�C�|)@S33��(����Ϳ�  C�:�                                    Bxk۱X  �          @p  @U�\?G�A?
=C��q@U���H��Q쿧�C�t{                                    Bxkۿ�  �          @n{@P  ��\)?G�AAp�C�� @P  �������
C��\                                    Bxk�Τ  �          @j=q@G
=��ff?.{A*�HC��3@G
=��33���R��z�C�H�                                    Bxk��J  �          @fff@@�׿���?+�A,z�C�<)@@�׿�Q쾮{��p�C��H                                    Bxk���  �          @g�@B�\��{?#�
A"�\C�L�@B�\��
=��p���{C���                                    Bxk���  �          @g�@<(�� ��?(��A(z�C��3@<(��z��(���33C�k�                                    Bxk�	<  �          @j�H@:�H�
=?5A2ffC��@:�H����G���p�C��
                                    Bxk��  
�          @l(�@?\)�33?+�A'�
C��q@?\)�
=��ff��\)C�Z�                                    Bxk�&�  �          @h��@5��
�H?=p�A;
=C�@ @5��  ��ff��33C���                                    Bxk�5.  �          @l��@6ff�G�?�RAp�C��q@6ff�G��(���HC���                                    Bxk�C�  �          @l��@8�����?.{A(��C�^�@8���\)��� ��C��                                    Bxk�Rz  �          @k�@=p��
=?\)A��C�E@=p��
=����C�J=                                    Bxk�a   �          @l(�@<���
=q>��@���C���@<���ff�+��(��C�AH                                    Bxk�o�  �          @j�H@5�{?��A  C��R@5��R����ffC���                                    Bxk�~l  �          @e@-p��p�?J=qAJ�HC�]q@-p���
��(���
=C��                                    Bxk܍  �          @^�R@#�
�
=q?fffAo�C��
@#�
��
���R��C���                                    Bxkܛ�  �          @qG�@/\)��?��
A|��C��
@/\)�#�
���R���RC�xR                                    Bxkܪ^  T          @l��@1G���?�=qA��RC���@1G����#�
�#33C�b�                                    Bxkܹ  
�          @dz�@#33��\?��
A�p�C�
=@#33�\)��=q���\C��\                                    Bxk�Ǫ  �          @s33@,���\)?xQ�Am��C���@,���(�þ�
=��=qC�Ф                                    Bxk��P  �          @j=q@
=�,��?8Q�A6{C��)@
=�,(��=p��;33C���                                    Bxk���  �          @l��@33�5�?#�
A=qC���@33�0�׿aG��]G�C�޸                                    Bxk��  �          @w
=@)���*�H?Y��AK�C�n@)���.�R��R��C�R                                    Bxk�B  �          @r�\@#�
�,��?333A+
=C��{@#�
�+��B�\�:ffC�˅                                    Bxk��  �          @q�@ ���1G�?
=A�HC�{@ ���+��fff�[�
C��f                                    Bxk��  �          @o\)@   �-p�?�RAC�T{@   �)���W
=�O33C���                                    Bxk�.4  �          @n�R@!��*�H?!G�A33C��{@!��'��Q��J{C���                                    Bxk�<�  �          @n�R@%��'
=?(��A"�\C�Y�@%��%��B�\�<  C��                                     Bxk�K�  �          @^�R@���{?�A�
C�@������L���UG�C�u�                                    Bxk�Z&  �          @a�@   ��H?��A�C���@   �Q�8Q��<  C�(�                                    Bxk�h�  �          @S33?�p��'
=        C��H?�p��\)��=q��\)C��)                                    Bxk�wr  �          @Q�@
=�!G��u���\C��{@
=��ÿ�=q���C�K�                                    Bxk݆  �          @Tz�@33�&ff����#�
C��@33�
=q������(�C���                                    Bxkݔ�            @U?�
=�.{���
��=qC��f?�
=�33���H����C��R                                    Bxkݣd  T          @]p�?��R�3�
>�@\)C��?��R��R������{C�U�                                    Bxkݲ
  �          @c33@33�*�H>�z�@�
=C�\)@33��Ϳ�\)����C���                                    Bxk���  
�          @^{@\)�(Q�>��R@��
C�0�@\)��H��=q���C�aH                                    Bxk��V  
�          @U�?˅�9��>�{@�=qC�S3?˅�*�H�������\C�^�                                    Bxk���  "          @X��?�{�<��>���@׮C�N?�{�/\)�����C�9�                                    Bxk��  �          @k�@��<(�?.{A+�C��q@��8Q�k��h��C��                                    Bxk��H  "          @hQ�?�
=�H��?(��A(��C��?�
=�B�\������C�o\                                    Bxk�	�  T          @^{?L���U�>��@���C���?L���Fff������C�0�                                    Bxk��  
�          @j=q@
=q�/\)?h��Aj�HC�q@
=q�4z��R��HC���                                    Bxk�':  
G          @o\)@�.{?��\A}p�C�XR@�7
=����C���                                    Bxk�5�  "          @h��@G��%�?�p�A�33C���@G��5��=q����C�Q�                                    Bxk�D�  "          @fff@!G���
?�Q�A�  C�� @!G��%�#�
�%�C��                                    Bxk�S,  �          @qG�@333�
�H?��A�z�C�3@333�!논#�
��C��                                    Bxk�a�  �          @tz�@1G��
�H?�Q�A��C���@1G��&ff=�?�G�C�ff                                    Bxk�px  �          @w
=@1����?\A��C�Ф@1��*�H>.{@!G�C�{                                    Bxk�  T          @w�@8���33?��A�\)C�Q�@8���#33>��@uC�L�                                    Bxkލ�  �          @z=q@5��Q�?��AƸRC�xR@5��*�H>���@�
=C�K�                                    Bxkޜj  �          @w
=@>{�޸R?�p�A��
C���@>{�Q�?
=Az�C��H                                    Bxkޫ  �          @w�@7���?���A���C��@7�� ��?&ffA33C�q�                                    Bxk޹�  �          @xQ�@:=q��z�?��HA�
=C�*=@:=q��?Q�AD(�C��                                    Bxk��\  �          @\)@5���@z�A��
C�޸@5��.{?G�A4  C�                                    Bxk��  "          @xQ�@&ff��(�@ffB�HC�b�@&ff�1G�?E�A8(�C��\                                    Bxk��  T          @p��@#33��\@ffBp�C���@#33�&ff?aG�AYC�/\                                    Bxk��N  �          @u�@1녿��@�RBp�C�R@1��33?��HA��\C�*=                                    Bxk��  �          @xQ�@Y���.{?�ffAޏ\C��
@Y������?���A�
=C�<)                                    Bxk��  �          @n{@N�R��?�A�=qC��@N�R��ff?��RA��C��                                    Bxk� @  �          @n�R@C�
��Q�@�B�C�+�@C�
��
=?�\A��C�xR                                    Bxk�.�  �          @q�@:�H    @��B�\C��R@:�H��  @33BQ�C�q�                                    Bxk�=�  T          @k�@+�>�G�@{B)\)A@+��h��@B��C���                                    Bxk�L2  T          @q�@6ff>#�
@{B#G�@K�@6ff���@(�B��C�                                    Bxk�Z�  	�          @r�\@2�\����@#33B(�
C���@2�\��{@ ��A�\)C��                                    Bxk�i~  "          @n�R@#�
���R@+�B8�C���@#�
��z�@Q�B�C���                                    Bxk�x$  �          @u@   ��33@8��BCp�C�f@   ��ff@�\BQ�C��                                    Bxk߆�  �          @w�@+����R@3�
B8z�C���@+���p�@�RB�C��q                                    Bxkߕp  "          @u@6ff��(�@#33B%�C���@6ff��Q�?���A�p�C���                                    Bxkߤ  
�          @z�H@8�þ�(�@'
=B'ffC��)@8�ÿ�p�@ ��A�ffC��f                                    Bxk߲�  �          @x��@0  >B�\@.{B2�@�G�@0  ��  @�HB{C���                                    Bxk��b  T          @`  @   <�@(�B1p�?!G�@   ��  @ffB�C���                                    Bxk��  
�          @]p�@�ý���@   B9\)C�Ǯ@�ÿ��@�B
=C�޸                                    Bxk�ޮ  
�          @W�@\)=L��@!G�BA=q?��@\)���\@�B ��C�C�                                    Bxk��T  �          @X��@>8Q�@(�B8�\@�
=@��\)@�B �C�8R                                    Bxk���  T          @J=q?��?Tz�@��BL�
A�  ?����@!�BU�C�%                                    Bxk�
�  T          @I��?�?
=q@�BJ�Az�R?��O\)@ffBB  C���                                    Bxk�F  �          @>�R?޸R���@z�BSffC��H?޸R��Q�?���B{C�=q                                    Bxk�'�  
�          @:�H?���>�Q�@   BlAb=q?�����  @z�BS�\C��{                                    Bxk�6�  �          @3�
?�{>�G�@{BTp�Av�H?�{�G�@�BG�C���                                    Bxk�E8  �          @,��?�  ?333@z�By{BG�?�  ��R@B}(�C�R                                    Bxk�S�  "          @-p�?O\)?@  @��B���B*�H?O\)�&ff@�RB���C���                                    Bxk�b�  �          @/\)?G�?��@
=BsQ�BV\)?G�����@$z�B�W
C�w
                                    Bxk�q*  �          @4z�?G�?k�@"�\B�B�BF��?G����@(��B�\)C�j=                                    Bxk��  �          @?\)?��?:�H@,(�B��)B
��?���J=q@+�B~��C��\                                    Bxk��v  �          @5�?h��?!G�@%B��)B
��?h�ÿQ�@"�\B�ffC��                                     Bxk��  �          @2�\���?�ff@��BY��BՏ\���>�@/\)B��C$W
                                    Bxk��  �          @/\)����?��@�Bi�RB��
���ý#�
@-p�B���C:xR                                    Bxk�h  T          @3�
<��
?�{@=qBr�\B�k�<��
��G�@1G�B�Q�C���                                    Bxk��  �          @7
=?J=q?��@�HBh(�Bi�H?J=q�8Q�@/\)B�
=C��
                                    Bxk�״  �          @8�ý���?�33@\)Br  B�B����ͽ�G�@6ffB��Cd�                                    Bxk��Z  
�          @9���aG�?���@"�\Bv�RBƊ=�aG��B�\@7�B��C\#�                                    Bxk��   T          @5�?Y��?���@��BoBP\)?Y������@'�B�u�C�l�                                    Bxk��  T          @H�ÿ���@�?�\)B�HB��
����?Q�@-p�BoG�C��                                    Bxk�L  T          @Dz´p�?�p�?�\B=qB�ff��p�?B�\@#33Bc��C��                                    Bxk� �  �          @DzῺ�H@z�?�B��B�ff���H?aG�@ ��B_=qC��                                    Bxk�/�  �          @H�ÿ�=q@33?�33A���B�=q��=q?��@'
=Bb��Cٚ                                    Bxk�>>  T          @H�ÿ��
@�?�B��B�ff���
?J=q@0  Bt�CY�                                    Bxk�L�  "          @Fff����?��H?�p�B�RB�녿���?�R@.{Bv�HC��                                    Bxk�[�  �          @G
=�\?�ff@G�B"  C+��\>��@*=qBlp�C"�)                                    Bxk�j0  �          @Fff���?�?��B
=C�����?0��@!�B[z�C33                                    Bxk�x�  �          @E����H@�?�\)BG�B�zῚ�H?G�@,(�Bw
=C5�                                    Bxk�|  �          @J=q�p��?���@��BF{B��p��>�\)@@  B���C#��                                    Bxk�"  �          @J=q�Y��?�  @{BO
=B���Y��>8Q�@AG�B���C'�                                    Bxk��  "          @C33�E�?�  @)��Bt{B��E���{@9��B���CK�R                                    Bxk�n  �          @B�\��{?��@(Q�Bo�C�׿�{��\@1�B���CL��                                    Bxk��  �          @C33����?��@%�BhC
�R�������@0  B��RCI5�                                    Bxk�к  �          @?\)��{?}p�@%Bp��C
33��{��@.{B��CM��                                    Bxk��`  
�          @@�׿��?O\)@+�B~=qC�3����:�H@,��B�CV�
                                    Bxk��  �          @?\)���\?333@-p�B�\C�)���\�W
=@*�HB~�C[�
                                    Bxk���  �          @:=q��(�>��R@&ffB�ffC%����(���{@�B\�
C^^�                                    Bxk�R  T          @7���  >aG�@$z�BQ�C*.��  ��
=@33BU
=C_ff                                    Bxk��  �          @:�H@�
?\?�Q�AǙ�B
=@�
?(��?���B 33A���                                    Bxk�(�  T          @C33?У�?�=q?���BQ�BA�R?У�?
=@ ��B]��A�=q                                    Bxk�7D  "          @H��?��
?�@�B@��B,
=?��
��@.�RBsG�C�xR                                    Bxk�E�  �          @HQ�?�Q�?333@\)B=��A��\?�Q�
=@G�BA  C�y�                                    Bxk�T�  �          @O\)?�?�p�@*�HBf(�B9�?���p�@:�HB�G�C�0�                                    Bxk�c6  5          @K�?�z�?+�@�\BP  A���?�z�&ff@�\BP��C�S3                                    Bxk�q�            @B�\@�\=L��?�33B�?�{@�\�u?��B{C��3                                    Bxk   T          @Dz�@33�B�\@   B#ffC��{@33��(�?˅A�
=C�                                      Bxk�(  
�          @C�
@��<�@�B/?O\)@�Ϳ���?�B�C��q                                    Bxk��  
�          @C�
@33>8Q�@��B;�@��@33���\?��HB"�C�Ф                                    Bxk�t  T          @H��@�?+�?�z�B33A|Q�@���ff?�p�B=qC���                                    Bxk�  
�          @W
=?}p�?��H@4z�Bg��B_{?}p����@J=qB�
=C�                                    Bxk���  "          @S�
?�p�?W
=@Q�B?��A�p�?�p����@p�BHz�C�33                                    Bxk��f  
�          @P��?�?��@&ffB^�A��R?��^�R@!G�BU33C�4{                                    Bxk��  A          @[�@(����R@33B,z�C�XR@(����R?��
B   C�O\                                    Bxk���  
�          @U�@*�H���
?�Q�Bp�C��
@*�H��ff?�p�AӅC���                                    Bxk�X  T          @N{@'
=�8Q�?�=qB�C��@'
=��\)?��HAمC�Z�                                    Bxk��  s          @H��@)���.{?�z�B 
=C�0�@)�����\?���A���C�o\                                    Bxk�!�  T          @K�@3�
�#�
?�p�A�C���@3�
�Q�?�p�A��RC��=                                    Bxk�0J  
�          @Mp�@5=u?��HA�=q?�(�@5�:�H?��\A�C�Ф                                    Bxk�>�  �          @L��@1G�>�{?��RA�R@�\@1G��   ?���A�\)C��                                     Bxk�M�  �          @L(�@C33�5>�A
�\C�n@C33�\(�<�>�C�'�                                    Bxk�\<  "          @K�@Fff�+�=�@	��C��)@Fff�!G�������C�7
                                    Bxk�j�  �          @Mp�@9�����\�#�
�5�C�.@9���}p��O\)�l(�C��\                                    Bxk�y�  �          @L��@:�H���    =uC��@:�H����.{�EC���                                    Bxk�.            @L(�@333���R�k���=qC�  @333��녿z�H��{C��=                                    Bxk��  
y          @L��@5���p��k���ffC�9�@5����׿z�H��G�C�'�                                    Bxk�z  �          @K�@333��p�����C��@333��Q�fff���C���                                    Bxk�   �          @O\)@0  ��Q콣�
����C�AH@0  ���׿xQ���p�C���                                    Bxk���  �          @Q�@.�R��=q���C�R@.�R��(������ffC���                                    Bxk��l  �          @N�R@0�׿�\)�\��Q�C��@0�׿���Q����\C��                                     Bxk��  �          @P  @&ff��z��G���(�C�� @&ff�Ǯ��{���HC��\                                    Bxk��  �          @Tz�@&ff�33�B�\�O\)C��{@&ff��녿�  ���C�޸                                    Bxk��^  �          @R�\@-p����#�
�0  C��@-p���(���{����C��                                     Bxk�  �          @QG�@�G�=���?�Q�C���@��(�������C���                                    Bxk��  �          @QG�@  �ff>aG�@s�
C��q@  �
=�������C�q�                                    Bxk�)P  �          @S33@�
�>�=q@��C�J=@�
�����\��=qC��\                                    Bxk�7�  �          @S�
@33�%>u@��HC�+�@33�zῗ
=��
=C��)                                    Bxk�F�  �          @U�?�  �4z�>W
=@eC��?�  �   ��=q��(�C�~�                                    Bxk�UB  �          @U�?��H�>�R�L�ͿY��C�?��H�   �У���p�C�"�                                    Bxk�c�  �          @U?�
=�6ff��Q���\)C�:�?�
=�{��=q�{C���                                    Bxk�r�  �          @Y��?��H�z�H?�A��C��?��H���׽#�
���RC�R                                    Bxk�4  �          @tz�?�
=?Ǯ@J=qB`��B>ff?�
=��Q�@`��B�z�C��3                                    Bxk��  �          @s33?^�R?�Q�@Q�BoQ�Bz?^�R���R@k�B���C��                                    Bxk䞀  �          @tz�?��?��
@O\)Bi=qBG{?����G�@c�
B�
=C���                                    Bxk�&  �          @u�?��\?���@R�\Bn�BB��?��\��@c�
B���C���                                    Bxk��  �          @u?z�H?��@S�
Bp(�Bl
=?z�H�\@j�HB���C�e                                    Bxk��r  �          @x��>�?У�@^{B��B��)>����@s33B��
C�޸                                    Bxk��  �          @w
=?\)?�z�@c33B��3B��H?\)�333@p  B�8RC�N                                    Bxk��  �          @y��?O\)?���@^{B|
=Bz��?O\)��@qG�B�G�C��                                     Bxk��d  �          @u?�z�?��\@^�RB�(�B%Q�?�z�}p�@^�RB��RC��{                                    Bxk�
  �          @w
=?��R=u@S33Bk�?�(�?��R���H@5�B<��C��                                     Bxk��  �          @xQ�?�<��
@Z=qBvp�?B�\?���ff@9��BA�C���                                    Bxk�"V  �          @z=q?���=#�
@XQ�Bp=q?�ff?�����\@8Q�B>=qC���                                    Bxk�0�  �          @z�H?��H=��
@X��Bo��@33?��H�޸R@:=qB?�C�0�                                    Bxk�?�  �          @}p�@G���Q�@N{BY�HC�q�@G���@!�B��C�!H                                    Bxk�NH  �          @�Q�@p��!G�@EBJ(�C�Ф@p���@�B
�RC�=q                                    Bxk�\�  �          @|��@�Ϳ��@7
=B<Q�C���@�����?�\)A�\C��=                                    Bxk�k�  �          @|��@�Ϳ�(�@"�\B!
=C�}q@���3�
?�p�A�  C��3                                    Bxk�z:  �          @\)@   ��\)@(Q�B%��C���@   �1G�?�{A�ffC��                                    Bxk��  �          @~{@Q��{@-p�B-=qC���@Q��333?�
=A�ffC�+�                                    Bxk嗆  	�          @w�@
=q��@(�B��C�g�@
=q�AG�?z�HAlQ�C�Ф                                    Bxk�,  "          @u@\)���
@5B>��C�R@\)�&ff?�(�A���C�^�                                    Bxk��  �          @w
=@=q���@,(�B0�HC�t{@=q�$z�?ǮA��RC���                                    Bxk��x  �          @tz�@���p�@#33B'�C�L�@��'
=?�\)A�C�q�                                    Bxk��  r          @u�@'���p�@ffB(�C�Ff@'�� ��?���A���C�!H                                    Bxk���  T          @w
=@>{��@ ��A�G�C�:�@>{�G�?p��Ab�\C�J=                                    Bxk��j  T          @xQ�@H�ÿ�
=?�\AٮC�� @H���
=q?=p�A/�C�                                    Bxk��  �          @xQ�@Z�H���
?�G�A�  C���@Z�H��G�>\@�z�C�g�                                    Bxk��  
�          @y��@Fff��p�?��HA��
C�!H@Fff�?�  Ap(�C��q                                    Bxk�\  �          @xQ�@=p�����@
=B�C���@=p��G�?���A�C�9�                                    Bxk�*  	�          @w�@B�\��=q@ffB=qC�4{@B�\��\?���A�G�C��                                    Bxk�8�  �          @xQ�@-p���z�@{B�HC�>�@-p��%?}p�An{C�%                                    Bxk�GN  �          @y��@*=q���@�B��C�U�@*=q�.�R?J=qA:�RC�(�                                    Bxk�U�            @xQ�@:=q��(�?�33A�RC��\@:=q�{?0��A#�
C��{                                    Bxk�d�  
          @x��@(�ÿ�\@�B	C��@(���*=q?fffAW�
C�\)                                    Bxk�s@  �          @x��@%��@  B��C�}q@%�.�R?n{A_33C���                                    Bxk��  �          @w�@*=q����@�B
=C�}q@*=q�$z�?�=qA��C��                                    Bxk搌  T          @w�@8Q쿽p�@�BC�j=@8Q��Q�?�  Ap��C�8R                                    Bxk�2  �          @w�@@  ��Q�?��HA�ffC�,�@@  ���?c�
AV=qC�xR                                    Bxk��  �          @w�@4z��z�?�
=A��C�  @4z�� ��>��@�z�C�4{                                    Bxk�~  "          @u�@;����H@�B��C�� @;��	��?�33A�C��
                                    Bxk��$  �          @u@>�R�xQ�@�B{C���@>�R��p�?�=qA���C�5�                                    Bxk���  �          @s33@<�Ϳ��@z�B�HC�ff@<���z�?��A���C�u�                                    Bxk��p            @tz�@9����Q�?޸RA�33C�� @9���
=?�A\)C�q�                                    Bxk��  r          @tz�@8�ÿУ�?�A�ffC�J=@8���
=?0��A(Q�C�aH                                    Bxk��  �          @u@1녿˅@�BQ�C�#�@1����?k�A_33C�H�                                    Bxk�b  "          @tz�@:�H=u@��B33?�=q@:�H����?��A��C�t{                                    Bxk�#  �          @xQ�@C�
?z�@�B�HA*�R@C�
�:�H@\)BC�T{                                    Bxk�1�  T          @x��@I���8Q�@��B=qC��@I����  ?�
=A�\)C�w
                                    Bxk�@T  T          @w�@.{��?��HA�  C�^�@.{�*�H?#�
A(�C��H                                    Bxk�N�  "          @tz�@)����?�(�A�
=C��\@)���-p�>���@�\)C�0�                                    Bxk�]�  T          @vff@0�׿���@��BffC��@0���   ?uAg\)C���                                    Bxk�lF  
�          @vff@p���=q@�B��C��=@p��0��?uAh��C��{                                    Bxk�z�  �          @u�@ �׿��
@G�B��C�N@ ���-p�?z�HAl��C�Z�                                    Bxk牒  T          @r�\@+���?���A���C�  @+��$z�?0��A)�C�)                                    Bxk�8  �          @qG�@/\)����@�
BC���@/\)�\)?��\A�
=C�^�                                    Bxk��  
�          @p  @�\� ��@�Bp�C�g�@�\�5�?:�HA5�C��                                    Bxk組  
�          @p��?�33��
@\)B3�C��?�33�Dz�?�G�A���C�J=                                    Bxk��*  "          @r�\�333���@5BB��C}�333�b�\?���A�G�C�k�                                    Bxk���  �          @s33���
=@<��BL��C������dz�?��RA��C��                                    Bxk��v  "          @s33=�Q����@<��BK�
C�)=�Q��fff?�p�A�\)C��)                                    Bxk��  �          @q�>���%�@0  B:�HC��H>���i��?uAk\)C�'�                                    Bxk���  T          @s33���/\)@'�B/33C��{���mp�?E�A<Q�C���                                    Bxk�h  
�          @p�׽L���.�R@$z�B-{C�|)�L���k�?:�HA3�
C���                                    Bxk�  
Z          @qG�<#�
�*=q@*=qB3C�"�<#�
�j�H?W
=AN�RC��                                    Bxk�*�  "          @l��<��
���@1G�BA�
C�J=<��
�b�\?�ffA���C�33                                    Bxk�9Z  �          @k�����&ff@%B3\)C�P�����e�?Q�AM�C��f                                    